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

def iterate_bboxes(image_width, image_height, tile_size):
    for y in range(0, image_height, tile_size):
        for x in range(0, image_width, tile_size):
            # Define bounding box coordinates
            bbox = (x, y, min(x + tile_size, image_width), min(y + tile_size, image_height))
            yield bbox

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

    pred = unet_predictions(frame_rescaled[np.newaxis,:,:],"lightsheet_2D_unet_root_ds1x",patch=[1,2048,2048])

    param_grid = {
        "beta": [ round(x,1) for x in np.arange(0.3,1.05,0.1)],
        "post_minsize": [ round(x,1) for x in np.arange(90,110,10)],
    }

    params = list(ParameterGrid(param_grid))

    for param in tqdm(params, desc="Post processing"):
        beta = param["beta"]
        post_mini_size = param["post_minsize"]

        mask = np.zeros_like(pred)
        image_width, image_height = pred.shape[1], pred.shape[2]
        tile_size = 4096
        
        # limited ram
        for bbox in tqdm(iterate_bboxes(image_width, image_height, tile_size), desc="Processing tiled watershed"):
            x0, y0, x1, y1 = bbox
            mask[x0:x1,y0:y1] = mutex_ws(pred[x0:x1,y0:y1],superpixels=None,beta=beta,post_minsize=post_mini_size,n_threads=6)
        mask_relab, fw, inv = relabel_sequential(mask[0,:,:])
        outlines = utils.masks_to_outlines(mask_relab)

        outX, outY = np.nonzero(outlines)
        img0 = image_to_rgb(frame_rescaled, channels=[0,0])
        imgout= img0.copy()
        imgout[outX, outY] = np.array([255,0,0]) # pure red

        # save report
        out_dir_ = os.path.join(args.output,"beta-{}_pms-{}".format(beta, post_mini_size))
        os.makedirs(out_dir_,exist_ok=True)
        OmeTiffWriter.save(img_dask,os.path.join(out_dir_,"img.ome.tif"),dim_order="YX")
        OmeTiffWriter.save(pred[0,:,:],os.path.join(out_dir_,"pred.ome.tif"),dim_order="YX")
        OmeTiffWriter.save(mask_relab,os.path.join(out_dir_,"mask.ome.tif"),dim_order="YX")
        OmeTiffWriter.save(imgout,os.path.join(out_dir_,"overlay.ome.tif"),dim_order="YXS")

if __name__ == "__main__":
    args = get_args()
    main(args)