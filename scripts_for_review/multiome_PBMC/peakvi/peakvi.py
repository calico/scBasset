import anndata
import scvi
import scanpy as sc
import h5py
import scipy
import pandas as pd

scvi.settings.verbosity = 40

ad = anndata.read_h5ad('/home/yuanh/sc_basset/datasets/10x_ARC_PBMC/ad_atac.h5ad')
scvi.data.setup_anndata(ad)
pvi = scvi.model.PEAKVI(ad)
pvi.train()
pvi.save("trained_model", overwrite=True)
