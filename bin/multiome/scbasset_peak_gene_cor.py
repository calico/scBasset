#!/usr/bin/env python

import h5py
import os
import glob
import sys
import subprocess

import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from optparse import OptionParser
from scbasset.utils import *

'''
scbasset_peak_gene_cor.py

filter candidate peaks by:
1. "peak-target-gene cor" > 0.4
2. "peak-target-gene cor" > "peak-other-nearby-gene cor"

Additional notes:
peak-gene cor: pearsonr across all expressed cells in dataset.
expressed cells: acc>0.1, log2rp10k>0.1.

This function takes as input:
    - scvi denoised ad_rna.
    - ad_atac.
    - trained scbasset model.
    - genome fasta.
    - query gene name.

and outputs:
    - peak-gene-cor heatmap (input folder).
    - filtered peaks and heatmap (output folder).

'''

def main():
    usage = 'usage: %prog [options] <anndata file> <trained model>'
    parser = OptionParser(usage)
    parser.add_option('--ad_rna', dest='ad_rna', help='RNA anndata.')
    parser.add_option('--ad_atac', dest='ad_atac', help='ATAC anndata.')
    parser.add_option('--model', dest='model', help='trained scBasset model h5.')
    parser.add_option('--fasta', dest='fasta', help='fasta file.')
    parser.add_option('--gene', dest='gene', help='gene name.')
    parser.add_option('--thres',  dest='thres', type="float", default=0.4, help='peak-gene correlation threshold.')
    parser.add_option('--input_dir', dest='indir', default='peaks', help='directory of peak_gene_nearby.R results.')
    parser.add_option('--output_dir', dest='outdir', default='peaks_filter', help='output directory.')
    
    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()
    
    # parse options
    ad_rna_file = options.ad_rna
    ad_atac_file = options.ad_atac
    trained_model = options.model
    fasta_file = options.fasta
    gene_name = options.gene
    cor_thres = options.thres
    input_dir = options.indir
    out_dir = options.outdir
    

    os.makedirs(out_dir, exist_ok=True)
    bed_file = '%s/peaks.bed'%input_dir    
    bed = pd.read_csv(bed_file, sep='\t', header=None)
    peak_names = bed.iloc[:,3]    
    ad = anndata.read_h5ad(ad_atac_file)

    ########################
    # compute denoised acc #
    ########################
    # write bed file
    seqs_dna, seqs_coords = make_bed_seqs_from_df(bed, fasta_file, seq_len=1344, stranded=False)
    peaks_1344 = pd.DataFrame(seqs_coords)
    peaks_1344.columns = ['chr','start','end']
    
    # write fasta file
    fa = open("%s/peaks.fasta"%input_dir, "w")
    for i in range(len(seqs_dna)):
        fa.write(">" + peak_names[i] + "\n" +seqs_dna[i] + "\n")
    fa.close()
    
    # compute denoised accessibility
    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model)
    accessibility = pred_on_fasta('%s/peaks.fasta'%input_dir, model, scale_method='sigmoid')
    accessibility = pd.DataFrame(accessibility, index=peak_names, columns=ad.obs_names).transpose()
    accessibility.to_csv('%s/denoised_acc.csv'%input_dir)
    
    #################
    # peak gene cor #
    #################    
    ad_rna = anndata.read_h5ad(ad_rna_file)
    genes = pd.read_csv('%s/genes_nearby.csv'%input_dir, index_col=0)
    dist_m = pd.read_csv('%s/peaks_gene_dist.csv'%input_dir, index_col=0)
    genes.index = genes['symbol']
    genes = genes.loc[genes.index.isin(ad_rna.var_names), :]
    log2rp10k = pd.DataFrame(np.log2(np.array(ad_rna[:,genes.index].layers['denoised'])/100+1), 
                             columns=genes.index, index=ad_rna.obs_names)
    log2rp10k = log2rp10k.loc[accessibility.index,:] # align RNA/ATAC cells

    # compute peak gene cor cor matrix
    m = pd.DataFrame(np.nan, 
                     columns=log2rp10k.columns, 
                     index=accessibility.columns)
    for i in accessibility.columns:
        for j in log2rp10k.columns:
            a = accessibility[i]
            b = log2rp10k[j]
            choose = (a>0.1) & (b>0.1)
            if (sum(choose)>100):
                m.loc[i, j] = pearsonr(a[choose], b[choose])[0]  
    m.index = accessibility.columns.str.cat(dist_m[gene_name].astype(str), sep=':')
    m.to_csv('%s/peak_gene_cor.csv'%input_dir)
    
    # plot
    toplot = m.transpose()
    f = sns.clustermap(toplot, col_cluster=False, row_cluster=False, 
                       figsize=(toplot.shape[1]/4+1, toplot.shape[0]/4+1), 
                       cmap='coolwarm', vmax=0.8, vmin=-0.8, dendrogram_ratio=0.1)
    f.savefig('%s/peak_gene_cor.png'%input_dir)

    ################
    # filter peaks #
    ################
    a = (m.idxmax(axis=1)==gene_name)
    b = (m[gene_name]>cor_thres)
    select = (a&b).values

    bed.loc[select,:].to_csv('%s/peaks.bed'%out_dir, index=False, header=False, sep='\t')
    bed['dist2tss'] = dist_m[gene_name].values
    bed.loc[select,:].to_csv('%s/peaks_annot.csv'%out_dir)

    toplot = m.loc[select,:].transpose()
    f = sns.clustermap(toplot, col_cluster=False, row_cluster=False,
                       figsize=(toplot.shape[1]/4+1, max(4, toplot.shape[0]/4+1)), 
                       cmap='coolwarm', vmax=0.8, vmin=-0.8, dendrogram_ratio=0.1)
    f.ax_heatmap.set_yticklabels(f.ax_heatmap.get_yticklabels(), rotation=0) # don't rotate y-label
    f.savefig('%s/peak_gene_cor.png'%out_dir)
    

if __name__ == '__main__':
    main()