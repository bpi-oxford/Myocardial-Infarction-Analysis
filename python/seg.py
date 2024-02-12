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
import pyvips
import tifffile
from ome_types.model import OME, Image, Pixels, Channel
from scipy import stats

def pyramidal_ome_tiff_write(image, path, resX=1.0, resY=1.0, units="µm", tile_size=2048, channel_colors=None):
    """
    Pyramidal ome tiff write is only support in 2D + C data.
    Input dimension order has to be XYC
    """

    assert len(image.shape) == 3, "Input dimension order must be XYC, get array dimension of {}".format(len(image.shape)) 

    size_x, size_y, size_c = image.shape
    
    format_dict = {
        np.uint8: "uchar",
        np.uint16: "ushort",
        np.float32: "float",
        np.float64: "double"
    }

    dtype_dict = {
        np.uint8: "uint8",
        np.uint16: "uint16",
        np.float32: "float",
        np.float64: "double"
    }

    if image.dtype not in list(format_dict.keys()):
        raise TypeError(f"Expected an uint8/uint16/float32/float64 image, but received {image.dtype}")

    im_vips = pyvips.Image.new_from_memory(image.transpose(1,0,2).reshape(-1,size_c).tobytes(), size_x, size_y, bands=size_c, format=format_dict[image.dtype.type]) 
    im_vips = pyvips.Image.arrayjoin(im_vips.bandsplit(), across=1) # for multichannel write
    im_vips.set_type(pyvips.GValue.gint_type, "page-height", size_y)

    # build minimal OME metadata
    ome = OME()

    if channel_colors is None:
        channel_colors = [-1 for _ in range(size_c)]

    img = Image(
        id="Image:0",
        name="resolution_1",
        pixels=Pixels(
            id="Pixels:0", type=dtype_dict[image.dtype.type], dimension_order="XYZTC",
            size_c=size_c, size_x=size_x, size_y=size_y, size_z=1, size_t=1, 
            big_endian=False, metadata_only=True,
            physical_size_x=resX,
            physical_size_x_unit=units,
            physical_size_y=resY,
            physical_size_y_unit=units,
            channels= [Channel(id=f"Channel:0:{i}", name=f"Ch_{i}", color=channel_colors[i]) for i in range(size_c)]
        )
    )

    ome.images.append(img)

    def eval_cb(image, progress):
        pbar_filesave.update(progress.percent - pbar_filesave.n)

    im_vips.set_progress(True)

    pbar_filesave = tqdm(total=100, unit="Percent", desc="Writing pyramidal OME TIFF", position=0, leave=True)
    im_vips.signal_connect('eval', eval_cb)
    im_vips.set_type(pyvips.GValue.gstr_type, "image-description", ome.to_xml())

    im_vips.write_to_file(
        path, 
        compression="lzw",
        tile=True, 
        tile_width=tile_size,
        tile_height=tile_size,
        pyramid=True,
        depth="onetile",
        subifd=True,
        bigtiff=True
        )

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
    SKIP_SEG=False
    BACKGROUND = 1200

    img = AICSImage(args.input)
    img.set_scene(SCENE_IDX)
    img_dask = img.get_image_dask_data("CYX")
    img_dask.persist()

    if not SKIP_SEG:
        # img_dask_rescaled = img_dask[CH_IDX,::SUBSAMPLE,::SUBSAMPLE]

        # # get modal value for background
        # bg = stats.mode(img_dask_rescaled, axis=None, keepdims=False)[0]
        # print("Detected modal intensity for background: {}".format(bg))

        # quick masking, only use for lower and upper percentile calculation
        img_dask_masked = img_dask[CH_IDX,:,:]
        pl, pu = np.percentile(np.ravel(img_dask_masked[img_dask_masked > BACKGROUND]), (TAIL, 100-TAIL))
        
        if not isinstance(pl, (int,float)):
            pl = pl.compute()
        if not isinstance(pu, (int,float)):
            pu = pu.compute()

        print("Percentiles: ({:.2f},{:.2f})".format(pl,pu))

        # segmentation
        print("Rescaling image intensity...")
        frame_rescaled = exposure.rescale_intensity(img_dask[CH_IDX,:,:], in_range=(pl, pu),out_range=(0,1))

        pred = unet_predictions(frame_rescaled[np.newaxis,:,:],"lightsheet_2D_unet_root_ds1x",patch=[1,2048,2048])

        # apply tissue mask for edge data
        pred_masked = pred[0,:,:]
        pred_masked[img_dask_masked < BACKGROUND] = 0

        # save report
        print("Saving segmentation labels...")
        os.makedirs(args.output,exist_ok=True)
        # pyramidal_ome_tiff_write(frame_rescaled.T[:,:,np.newaxis], os.path.join(args.output,"img.ome.tif"), resX=img.physical_pixel_sizes.X, resY=img.physical_pixel_sizes.Y)
        pyramidal_ome_tiff_write(pred_masked.T[:,:,np.newaxis].astype(np.float32), os.path.join(args.output,"pred.ome.tif"), resX=img.physical_pixel_sizes.X, resY=img.physical_pixel_sizes.Y)
    
    if SKIP_SEG:
        print("Skipping segmentation, load label directly")
        pred = tifffile.imread(os.path.join(args.output,"pred.ome.tif")).T[np.newaxis,:,:]

    # param_grid = {
    #     "beta": [ round(x,1) for x in np.arange(0.3,1.05,0.1)],
    #     "post_minsize": [ round(x,1) for x in np.arange(90,110,10)],
    # }

    # params = list(ParameterGrid(param_grid))

    # image_width, image_height = img_dask.shape[1], img_dask.shape[2]
    # tile_size = 4096
    
    # bboxes = iterate_bboxes(image_width, image_height, tile_size)
    # bboxes = [bbox for bbox in bboxes]

    # for param in tqdm(params, desc="Post processing"):
    #     beta = param["beta"]
    #     post_mini_size = param["post_minsize"]

    #     # limited ram
    #     mask = np.zeros_like(pred[0,:,:])
    #     for bbox in tqdm(bboxes,total=len(bboxes), desc="Processing tiled watershed"):
    #         x0, y0, x1, y1 = bbox
    #         pred_ = pred[0,x0:x1,y0:y1]
    #         mask[x0:x1,y0:y1] = mutex_ws(pred_,superpixels=None,beta=beta,post_minsize=post_mini_size,n_threads=6)
    #     mask_relab, fw, inv = relabel_sequential(mask[0,:,:])
    #     outlines = utils.masks_to_outlines(mask_relab)

    #     outX, outY = np.nonzero(outlines)
    #     img0 = image_to_rgb(frame_rescaled, channels=[0,0])
    #     imgout= img0.copy()
    #     imgout[outX, outY] = np.array([255,0,0]) # pure red

    #     # save watershed results
    #     out_dir_ = os.path.join(args.output,"beta-{}_pms-{}".format(beta, post_mini_size))
    #     os.makedirs(out_dir_,exist_ok=True)
    #     OmeTiffWriter.save(mask_relab,os.path.join(out_dir_,"mask.ome.tif"),dim_order="YX")
    #     OmeTiffWriter.save(imgout,os.path.join(out_dir_,"overlay.ome.tif"),dim_order="YXS")

if __name__ == "__main__":
    args = get_args()
    main(args)