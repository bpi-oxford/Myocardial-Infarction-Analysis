import os
import argparse
from plantseg.predictions.functional.predictions import unet_predictions
from plantseg.segmentation.functional.segmentation import *
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from skimage import exposure
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops, regionprops_table
from sklearn.model_selection import ParameterGrid
from skimage.segmentation import relabel_sequential
from cellpose import core, utils, io, models, metrics, plot
from cellpose.plot import *
import pandas as pd

def is_valid_file_or_directory(path):
    """Check if the given path is a valid file or directory."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Path '{path}' does not exist.")
    return path

def get_args():
    parser = argparse.ArgumentParser(prog="decon",
                            description="WSI color deconvolution tool")
    parser.add_argument(
        "-i", "--input", 
        dest="input",
        help="Path to the input CZI file",
        metavar="PATH",
        type=is_valid_file_or_directory,
        required=True
        )
    parser.add_argument(
        "-o", "--output", 
        dest="output",
        help="Path to the output directory",
        metavar="DIR",
        required=True
        )

    return parser.parse_args()

def main(args):
    SCENE_IDX=1
    CH_IDX=1
    SUBSAMPLE=10
    TAIL=1

    img = AICSImage(args.input)
    img.set_scene(SCENE_IDX)
    img_dask = img.get_image_dask_data("CYX")
    img_dask.persist()

    img_dask_rescaled = img_dask[CH_IDX,::SUBSAMPLE,::SUBSAMPLE]
    pl, pu = np.percentile(np.ravel(img_dask_rescaled), (TAIL, 100-TAIL))
    
    if not isinstance(pl, (int,float)):
        pl = pl.compute()
    if not isinstance(pu, (int,float)):
        pu = pu.compute()

    # segmentation
    print("Rescaling image intensity...")
    frame_rescaled = exposure.rescale_intensity(img_dask, in_range=(pl, pu),out_range=(0,1))

    pred = unet_predictions(frame_rescaled[np.newaxis,:,:],"lightsheet_2D_unet_root_ds1x",patch=[1,1024,1024])

    res = []
    param_grid = {
        "beta": [ round(x,1) for x in np.arange(0.3,1.05,0.1)],
        "post_minsize": [ round(x,1) for x in np.arange(90,110,10)],
    }

    params = list(ParameterGrid(param_grid))

    for param in tqdm(params, desc="Post processing"):
        beta = param["beta"]
        post_mini_size = param["post_minsize"]

        mask = mutex_ws(pred,superpixels=None,beta=beta,post_minsize=post_mini_size,n_threads=6)
        mask_relab, fw, inv = relabel_sequential(mask[0,:,:])
        outlines = utils.masks_to_outlines(mask_relab)

        outX, outY = np.nonzero(outlines)
        img0 = image_to_rgb(frame_rescaled, channels=[0,0])
        imgout= img0.copy()
        imgout[outX, outY] = np.array([255,0,0]) # pure red

        # perform region properties measurements
        props = regionprops(mask_relab)
        props_table = regionprops_table(mask_relab,properties=["centroid","area","area_convex","area_filled","axis_major_length","axis_minor_length","eccentricity","feret_diameter_max","perimeter","perimeter_crofton","solidity"])
        props_df = pd.DataFrame.from_dict(props_table)

        res.append({
            "beta": beta,
            "post_mini_size": post_mini_size,
            "img": img_dask,
            "pred": pred[0,:,:],
            "mask": mask_relab,
            "overlay": imgout,
            "props": props,
            "props_df": props_df
        })

    # save report
    os.makedirs(args.output,exist_ok=True)
    for res_ in tqdm(res,desc="Result saving"):
        out_dir_ = os.path.join(args.output,"beta-{}_pms-{}".format(res_["beta"], res_["post_mini_size"]))
        os.makedirs(out_dir_,exist_ok=True)
        OmeTiffWriter.save(res_["img"],os.path.join(out_dir_,"img.ome.tif"),dim_order="YX")
        OmeTiffWriter.save(res_["pred"],os.path.join(out_dir_,"pred.ome.tif"),dim_order="YX")
        OmeTiffWriter.save(res_["mask"],os.path.join(out_dir_,"mask.ome.tif"),dim_order="YX")
        OmeTiffWriter.save(res_["overlay"],os.path.join(out_dir_,"overlay.ome.tif"),dim_order="YXS")
        res_["props_df"].to_csv(os.path.join(out_dir_,"props.csv"),index=False)

if __name__ == "__main__":
    args = get_args()
    main(args)