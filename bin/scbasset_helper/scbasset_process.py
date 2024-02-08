import os
import scipy
import anndata
import glob
import re
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scbasset.utils import *
from optparse import OptionParser

'''
scbasset_process.py
This function takes as input:
    - the preprocessed anndata (from scbasset_preprocess.py);
    - the scbasset trained model (from scbasset_train.py);
    - 
and outputs:
    - intercepts vs. sequencing depth correlation plot.
    - a new anndata file with scbasset embeddings in ad.obsm['scbasset']. umap computed based on scbasset embeddings.
    - 
'''

def main():
    usage = 'usage: %prog [options] <anndata file> <trained model> <motif folder>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='outdir', default='results',
                      help='output directory')
    
    (options, args) = parser.parse_args()
    if len(args) != 3:
        parser.error('must provide 3 arguments as inputs.')
    else:
        ad_file = args[0]
        trained_model = args[1]
        motif_fasta_folder = args[2]

    # parse options
    outdir = options.outdir
    
    # load weights
    os.makedirs(outdir, exist_ok=True)
    ad = anndata.read_h5ad(ad_file)
    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model)
    
    # plot intercept vs. depth    
    sc.pp.filter_cells(ad, min_genes=0)
    intercept = get_intercept(model) # get_intercept function
    f, ax = plt.subplots(figsize=(4,4))
    r = scipy.stats.pearsonr(intercept, np.log10(ad.obs['n_genes']))[0]
    sns.scatterplot(x=intercept, y=np.log10(ad.obs['n_genes']), ax=ax)
    ax.set_xlabel('intercept')
    ax.set_ylabel('log10(n_peaks)')
    ax.set_title('Pearson R: %.3f'%r)
    f.savefig('%s/intercept.pdf'%outdir)
    
    # save cell embeddings
    proj = get_cell_embedding(model) # get_cell_embedding function
    pd.DataFrame(proj).to_csv('%s/projection_atac.csv'%outdir)
    ad.obsm['scbasset'] = proj
    sc.pp.neighbors(ad, use_rep='scbasset')
    sc.tl.umap(ad)
    ad.write('%s/ad_scbasset.h5ad'%outdir)
    
    # score motifs
    tfs = sorted([re.sub('.*/|.fasta','',i) for i in glob.glob('%s/shuffled_peaks_motifs/*'%motif_fasta_folder)])
    out = pd.DataFrame(np.nan, columns=tfs, index=ad.obs_names)
    for i in tfs:
        out.loc[:,i] = motif_score(i, model, motif_fasta_folder)
        print(i)
    out.to_csv('%s/motif_scores.csv'%outdir)
    
if __name__ == '__main__':
    main()