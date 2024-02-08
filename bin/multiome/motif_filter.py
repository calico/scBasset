#!/usr/bin/env python

import numpy as np
import scanpy as sc
import pandas as pd
import anndata
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
from optparse import OptionParser

def dedup(df, cols):
    tmp = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return df.loc[~tmp.duplicated(),:]

def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('--ad_rna', dest='ad_rna', help='RNA anndata.')
    parser.add_option('--ad_atac', dest='ad_atac', help='ATAC anndata.')
    parser.add_option('--motif2tf', dest='motif2tf', help='meme motif to tf mapping file.')
    parser.add_option('--gene', dest='gene', help='gene name.')
    parser.add_option('--ism_tfrna_thres',  dest='ism_tfrna_thres', type="float", default=0.2, help='ism-tfrna-cor threshold.')
    parser.add_option('--ism_targetrna_thres',  dest='ism_targetrna_thres', type="float", default=0.2, help='ism-targetrna-cor threshold.')
    parser.add_option('--input_dir', dest='indir', default='out', help='directory of scbasset_peak_ism.py results.')
    parser.add_option('--peak_annot', dest='peak_annot', 
                      default='peaks_filter/peaks_annot.csv', 
                      help='peak annotation file. Output of scbasset_peak_gene_cor.py.')    
    
    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()
    
    # parse options
    ad_rna_file = options.ad_rna
    ad_atac_file = options.ad_atac
    motif2tf_file = options.motif2tf
    gene_name = options.gene
    ism_tfrna_thres = options.ism_tfrna_thres
    ism_targetrna_thres = options.ism_targetrna_thres
    input_dir = options.indir
    peak_annot_file = options.peak_annot
    
    ad = anndata.read_h5ad(ad_atac_file)
    ad_rna = anndata.read_h5ad(ad_rna_file)
    motif2tf = pd.read_csv(motif2tf_file, index_col=0)
    peak_annot = pd.read_csv(peak_annot_file, index_col=0)

    # motif hit matrix
    ism_m = np.load('%s/ism_matrix.npy'%input_dir)
    hits = pd.read_csv('%s/motif_hits.csv'%input_dir, index_col=0)
    motif2tf.set_index('motif_names', inplace=True)
    hits['tf'] = motif2tf.loc[hits['motif_id'],'tfs'].values

    #############################
    # save group-by-hits matrix #
    #############################
    a = pd.DataFrame(ism_m).transpose()
    a['group'] = ad.obs['group'].values
    a = a.groupby('group').mean().transpose()
    a.index = hits.index
    a.to_csv('%s/motif_by_group_ism.csv'%input_dir)

    ###################
    # add ism-RNA cor #
    ###################
    # ism-tf_mrna correlation
    # ism-target_mrna correlation
    ad_rna.X = np.log2(ad_rna.layers['denoised']/100+1)
    acc = pd.read_csv('%s/desnoised_acc.csv'%input_dir, index_col=0)
    hits['ism_tfmrna_cor'] = np.nan
    hits['ism_targetmrna_cor'] = np.nan

    for i in range(hits.shape[0]):
        hit_id = hits.index[i]
        tf = hits.loc[hit_id, 'tf']
        peak_id = hits.loc[hit_id, 'sequence_name']
        if tf in ad_rna.var_names:
            a = np.array(ad_rna[:,tf].X).flatten()
            b = ism_m[i,:]
            select = (acc[peak_id]>0.1) & (a>0.1)
            if sum(select)>=10: hits.loc[hit_id, 'ism_tfmrna_cor'] = pearsonr(a[select], b[select])[0]
        a = np.array(ad_rna[:,gene_name].X).flatten()
        b = ism_m[i,:]
        select = (acc[peak_id]>0.1) & (a>0.1)
        hits.loc[hit_id, 'ism_targetmrna_cor'] = pearsonr(a[select], b[select])[0]

    cols = ['motif_id', 'tf', 'sequence_name', 'start', 'stop', 'strand', 'p-value', 'matched_sequence', 'dot', 
            'dot_qval', 'ism', 'ism_qval', 'ism_tfmrna_cor', 'ism_targetmrna_cor']
    hits = hits.loc[:, cols]
    hits.to_csv('%s/motif_hits_rna.csv'%input_dir)

    ##########
    # filter #
    ##########
    # by ism_tfmrna_cor
    # by ism_targetmrna_cor
    hits_filter = hits.loc[(hits['ism_tfmrna_cor']>ism_tfrna_thres
                           ) & (hits['ism_targetmrna_cor']>ism_targetrna_thres
                               ),:]
    
    hits_filter = dedup(hits_filter, ['sequence_name', 'tf', 'start'])
    hits_filter = hits_filter.sort_values('start')
    hits_filter.to_csv('motif_hits_rna_filter.csv')

    ############
    # overview #
    ############
    f, axs = plt.subplots(figsize=(14, 3), ncols=3)
    toplot1 = hits_filter['tf'].value_counts()[:20]
    toplot2 = hits_filter.loc[:,['tf','dot']].groupby('tf').mean().sort_values('dot', ascending=False)[:20]
    toplot3 = hits_filter.loc[:,['tf', 'dot']].groupby('tf').sum().sort_values('dot', ascending=False)[:20]

    toplot1.plot.bar(ax=axs[0])
    toplot2.plot.bar(ax=axs[1])
    toplot3.plot.bar(ax=axs[2])
    axs[0].set_title('sort by motif occurence')
    axs[0].set_ylabel('motif frequency')
    axs[1].set_title('sort by mean motif score (pwm-ism dot)')
    axs[1].set_ylabel('mean of motif score')
    axs[2].set_title('sort by total motif score')
    axs[2].set_ylabel('sum of motif score')
    f.tight_layout()
    f.autofmt_xdate(rotation=45)
    f.savefig('tf_overview.png')

if __name__ == "__main__":
    main()
