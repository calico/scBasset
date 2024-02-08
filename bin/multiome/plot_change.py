#!/usr/bin/env python

import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pysam
from scipy.stats import pearsonr
from optparse import OptionParser


def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('--ad_rna', dest='ad_rna', help='RNA anndata.')
    parser.add_option('--ad_atac', dest='ad_atac', help='ATAC anndata.')
    parser.add_option('--gene', dest='gene', help='gene name.')

    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()
    
    # parse options
    ad_rna_file = options.ad_rna
    ad_atac_file = options.ad_atac
    gene_name = options.gene
    acc_file = 'out/desnoised_acc.csv'
    peak_annot_file = 'peaks_filter/peaks_annot.csv'

    ###################
    # plot expression #
    ###################
    ad_rna = anndata.read_h5ad(ad_rna_file)
    ad_rna.obs['celltype'] = ad_rna.obs['celltype_sex1'].str.replace(' .*','', regex=True)
    ad_rna.obs['treatment'] = ad_rna.obs['treatment'].cat.reorder_categories(['CAS', 'AD'])
    toplot = ad_rna.obs.copy()
    toplot['gene'] = np.log2(np.array(ad_rna[:, gene_name].layers['denoised']).flatten()/100+1)

    cts = sorted(ad_rna.obs['celltype'].unique())
    f, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x='celltype', y='gene', hue='treatment', data=toplot, ax=ax, order=cts)
    ax.set_ylabel('%s (log2RP10K)'%gene_name)
    ax.set_xlabel('')
    #axs[i].axhline(y=0.5, color='grey', linestyle='--')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    f.tight_layout()
    f.savefig('gene_expression.pdf')

    ###################
    # plot acc change #
    ###################
    ad_atac = anndata.read_h5ad(ad_atac_file)
    acc = pd.read_csv(acc_file, index_col=0)
    acc['celltype'] = ad_atac.obs['celltype']
    ad = acc.loc[ad_atac.obs['treatment']=='AD',:]
    cas = acc.loc[ad_atac.obs['treatment']=='CAS',:]
    acc_dif = ad.groupby('celltype').mean() - cas.groupby('celltype').mean()

    # add dist2tss
    annot = pd.read_csv(peak_annot_file, index_col=0)
    annot.index = annot.iloc[:,3]
    acc_dif.columns = acc_dif.columns.str.cat(annot.loc[acc_dif.columns,'dist2tss'].astype(str), sep=':')

    f = sns.clustermap(acc_dif.transpose(), col_cluster=False, row_cluster=False, 
                       figsize=(8, 5), 
                       cmap='coolwarm', vmax=0.5, vmin=-0.5, dendrogram_ratio=0.1)
    f.ax_heatmap.set_title('acc change (AD-CAS)')
    f.ax_heatmap.set_xticklabels(f.ax_heatmap.get_xticklabels(),rotation=45)
    f.savefig('acc_change.pdf')    
    
if __name__ == '__main__':
    main()