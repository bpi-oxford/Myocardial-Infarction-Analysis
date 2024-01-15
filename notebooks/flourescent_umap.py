# %% [markdown]
# # UMAP Segmentation for the Fluorscent Imaging

# %% [markdown]
# For image segmentation is a computational heavy task. Recommend to use GPU version of UMAP for fast output.
# 
# Key package dependency:
# - [cuml](https://docs.rapids.ai/install)
# - [umap](https://umap-learn.readthedocs.io/en/latest/index.html)
# - [numba](https://numba.pydata.org/numba-doc/latest/index.html)
# - [dask-cuda](https://docs.rapids.ai/api)


# %%
import os
from aicsimageio import AICSImage
import matplotlib.pyplot as plt

import dask
import dask.array as da
import dask.array.image
from dask.distributed import Client, progress, Scheduler, Worker, Nanny, SpecCluster
from dask_cuda import CUDAWorker
from dask_cuda.worker_spec import worker_spec
from dask_cuda.initialize import initialize
import psutil
import multiprocessing

from pprint import pprint
from xml.etree.ElementTree import ElementTree, Element
import json
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
import umap

import cuml
import cupy as cp

def main():

    # %%
    # gather device info
    cpu_count = multiprocessing.cpu_count()
    memory_count = psutil.virtual_memory().total
    print("CPU count:", cpu_count)
    print("System memory:",round(memory_count/(1024*1024*1024),2), "GB")

    # %%
    specs = {
        "cpu":{
            "scale":3,
            "resources":{
            }
        },
        "gpu":{
            "scale":1,
            "resources":{
                "CUDA_VISIBLE_DEVICES": [0],
            }
        }
    }

    worker_count = 0
    for v in specs.values():
        worker_count += v["scale"]

    nthreads = cpu_count//worker_count
    memory_limit = int(memory_count*0.9)//worker_count # set to use 90% of the system memory to avoid crashing

    print("number of workers:", worker_count)
    print("threads per worker:", nthreads)
    print("memory limit per worker:", round(memory_limit/(1024*1024*1024),2), "GB")

    # %%
    workers = {}

    for k, v in specs.items():
        for i in range(v["scale"]):
            if "CUDA_VISIBLE_DEVICES" in v["resources"].keys():
                workers["{}-{}".format(k,i)] = worker_spec(
                    threads_per_worker=nthreads, 
                    CUDA_VISIBLE_DEVICES=v["resources"]["CUDA_VISIBLE_DEVICES"]
                    )[0]
                workers["{}-{}".format(k,i)]["options"]["resources"]={"GPU":len(v["resources"]["CUDA_VISIBLE_DEVICES"])}
                workers["{}-{}".format(k,i)]["options"]["memory_limit"]=memory_limit
            else:
                workers["{}-{}".format(k,i)] = {
                    "cls":Nanny,
                    "options":{
                        "nthreads": nthreads,
                        "memory_limit": memory_limit
                        }
                }     
                
    workers

    # %%
    scheduler = {'cls': Scheduler, 'options': {"dashboard_address": ':8787'}}
    cluster = SpecCluster(scheduler=scheduler, workers=workers)
    client = Client(cluster)
    client

    # %% [markdown]
    # ## Data Loading

    # %% [markdown]
    # For effective data loading, use [aicsimageio](https://github.com/AllenCellModeling/aicsimageio)

    # %%
    # idrive
    # IMAGE_DATA = "/mnt/Imaging/Group Fritzsche/Jacky/myocardial cells/For sharing analysis/no097_FR_D30_Q2_lv2_CD31-AF555_PDGFRa-AF647_WGA-AF488.czi"

    # kirpc541
    IMAGE_DATA = "/home/jackyko/Jacky/data/mycardial_cells/no097_FR_D30_Q2_lv2_CD31-AF555_PDGFRa-AF647_WGA-AF488.czi"

    # %%
    img = AICSImage(IMAGE_DATA)
    print(img.dims.order)  # T, C, Z, Y, X, (S optional)
    img.dask_data

    # %%
    img.scenes

    # %%
    img.set_scene("ScanRegion0")

    # %% [markdown]
    # For faster display, we load the image with subsampling

    # %%
    SCALE = 10

    # %%
    img_dask_sub = img.dask_data[:,:,::SCALE,::SCALE,::SCALE]
    img_dask_sub

    # %% [markdown]
    # ## Channel Metadata

    # %%
    def etree_to_dict(t):
        d = {t.tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v
                        for k, v in dd.items()}}
        if t.attrib:
            d[t.tag].update(('@' + k, v)
                            for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[t.tag]['#text'] = text
            else:
                d[t.tag] = text
        return d

    # %%
    META = img.metadata

    # %%
    d = etree_to_dict(META)

    pprint(d)

    # %%
    # image information metadata
    channel_info = d["ImageDocument"]["Metadata"]["Information"]["Image"]['Dimensions']['Channels']['Channel']
    pprint(channel_info)


    # %% [markdown]
    # ## Display the Image

    # %%
    fig, axs = plt.subplots(1,4,figsize=(10,10))

    for i in range(4):
        ax = axs[i]
        ax.set_axis_off()
        img_ch = img_dask_sub[0,i,0,]
        lp = da.percentile(img_ch.ravel(),0.5)
        up = da.percentile(img_ch.ravel(),99.5)
        ax.imshow(img_ch,cmap="gray",vmin=lp,vmax=up)
        ax.set_title(channel_info[i]['@Name'])

    # %% [markdown]
    # ## Pixel Intensity Clustering for Region Identification, using UMAP

    # %% [markdown]
    # Convert the multi channel image to vector form before applying UMAP projection

    # %%
    input = img_dask_sub.T.reshape(-1,4)/255.0
    input

    # kmeans_multi = MiniBatchKMeans(4)
    # kmeans_multi.fit(input)
    # new_colors = kmeans_multi.cluster_centers_[kmeans_multi.predict(input)]

    # # plot_pixels(input, colors=new_colors,title="Reduced color space: 3 colors")
    # fig, axs = plt.subplots(1,4,figsize=(20,5))
    # labels = kmeans_multi.labels_.reshape(images[1]['data'].shape[1:3])
    # for i,ax in enumerate(axs):
    #     ax.set_title("Region {}".format(i))
    #     ax.imshow(labels==i,cmap="gray")
    #     ax.set_axis_off()

    # %% [markdown]
    # ### Normalize Input Data

    # %%   %%time
    # cpu version
    scaled_input_data_cpu = StandardScaler().fit_transform(input)

    # %% [markdown]
    # It is advised to use all workers to transfer data to GPU, then perform calculation on single gpu worker.

    # %%  %%time
    input_gpu = cp.array(input)

    # %%  %%time
    # gpu version
    with dask.annotate(resources={'GPU': 1}):
        scaled_input_data_gpu = cuml.preprocessing.StandardScaler().fit_transform(input_gpu)

    # %% [markdown]
    # ### CPU UMAP fitting

    # %%time
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        n_epochs=10,
        min_dist=0.1,    
        random_state=42,
        verbose=True
    )

    embedding_cpu = reducer.fit_transform(scaled_input_data_cpu, verbose=True)
    embedding_cpu.shape

    # %%
    # df = pd.DataFrame(embedding, columns=('x', 'y'))
    # df['class'] = pd.Series([str(x) for x in target], dtype="category")

    # cvs = ds.Canvas(plot_width=400, plot_height=400)
    # agg = cvs.points(df, 'x', 'y', ds.count_cat('class'))
    # img = tf.shade(agg, color_key=color_key, how='eq_hist')

    # utils.export_image(img, filename='fashion-mnist', background='black')

    # image = plt.imread('fashion-mnist.png')
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plt.imshow(image)
    # plt.setp(ax, xticks=[], yticks=[])
    # plt.title("Fashion MNIST data embedded\n"
    #           "into two dimensions by UMAP\n"
    #           "visualised with Datashader",
    #           fontsize=12)

    # plt.show()

    # %%
    plt.scatter(
        embedding_cpu[:, 0],
        embedding_cpu[:, 1],
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the fluorescent intensity on CPU')
    plt.savefig("./cpu_umap.png")

    # %% [markdown]
    # ### GPU UMAP fitting

    # %%time
    reducer = cuml.UMAP(
        n_neighbors=15,
        n_components=2,
        n_epochs=10,
        min_dist=0.1,
        verbose=True
    )
    embedding_gpu = reducer.fit_transform(scaled_input_data_gpu)

    # %%
    # df_sample = df.sample(frac=0.25)
    # fig, ax = plt.subplots(1, figsize=(14, 10))
    # plt.scatter(
    #     df_sample["x"].values, df_sample["y"].values, 
    #     c=df_sample["class"].astype("int"), s=0.3, cmap='Spectral', alpha=1.0)
    # plt.setp(ax, xticks=[], yticks=[])
    # cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    # cbar.set_ticks(np.arange(10))
    # plt.title('Fashion MNIST Embedded via UMAP')

    # %%
    plt.scatter(
        embedding_gpu.get()[:, 0],
        embedding_gpu.get()[:, 1],
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the fluorescent intensity on GPU')
    plt.savefig("./gpu_umap.png")

    # %% [markdown]
    # ### Traditional Clustering
    # Follow the instruction of the UMAP documentation: https://umap-learn.readthedocs.io/en/latest/clustering.html#traditional-clustering


if __name__ == "__main__":
    main()