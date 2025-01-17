#!/usr/bin/env python
import h5py
import os
import glob
import sys
import subprocess

import numpy as np
import scanpy as sc
import pandas as pd
from scipy.stats import pearsonr
from optparse import OptionParser
from scbasset.utils import *

from Bio.Seq import Seq


def motif_ism(new_model, w, seq, start, end):
    # given a scbasset model, and final layer weight
    # predict accessibility of the seq
    # compute ism for a range of nucleotides
    
    # output tensor
    m = np.zeros((w.shape[1], end-start, 4))
    
    # reference sequence
    seq_ref_1hot = dna_1hot(seq)
    seqs_1hot_tf = tf.convert_to_tensor(seq_ref_1hot, dtype=tf.float32)[tf.newaxis]
    latent_ref = new_model(seqs_1hot_tf)
    latent_ref = tf.squeeze(latent_ref, axis=[0,1])
    pred_ref = tf.einsum('b,bt->t', latent_ref, w)

    for i in range(start, end):
        out = []
        for j in range(4):
            tmp = np.copy(seq_ref_1hot)
            tmp[i,:] = [False, False, False, False]
            tmp[i,j] = True
            out += [tmp]
        
        out_tf = tf.convert_to_tensor(np.array(out), dtype=tf.float32)
        latent = new_model(out_tf)
        latent = tf.squeeze(latent, axis=[1])
        latent = latent - latent_ref # ism on latent space

        pred = tf.einsum('nb,bt->nt', latent, w)
        m[:,i-start,:] = pred.numpy().transpose()
        
    return pred_ref.numpy(), m

'''
target_search.py


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
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('--tf_id', dest='tf_id', help='TF ID.')
    parser.add_option('--tf_name', dest='tf_name', help='TF name.')
    parser.add_option('--ad_rna', dest='ad_rna', help='RNA anndata.')
    parser.add_option('--ad_atac', dest='ad_atac', help='ATAC anndata.')
    parser.add_option('--ad_tf', dest='ad_tf', help='TF anndata.')
    parser.add_option('--model', dest='model', help='trained scBasset model h5.')
    parser.add_option('--fasta', dest='fasta', help='peak sequences file in fasta format.')
    parser.add_option('--fimo', dest='fimo', help='fimo.tsv FIMO output file.')
    parser.add_option('--corr_file', dest='corr_file', help='peak-gene corr file.')
    parser.add_option('--pwm_path', dest='pwm_path', help='pwm file path.')
    parser.add_option('--thres',  dest='thres', type="float", default=0.6, help='peak-gene correlation threshold.')
    parser.add_option('--output_dir', dest='outdir', default='ism_out', help='output directory.')
    
    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()

    # input files
    motif_id = options.tf_id
    tf_name = options.tf_name
    ad_atac_file = options.ad_atac
    ad_rna_file = options.ad_rna
    ad_tf_file = options.ad_tf
    model_file = options.model
    fasta_file = options.fasta
    fimo_file = options.fimo
    corr_file = options.corr_file
    pwm_path = options.pwm_path
    peak_gene_thres = options.thres
    outdir = options.outdir

    # create output directory    
    os.makedirs(outdir, exist_ok=True)

    ad = anndata.read_h5ad(ad_atac_file)
    ad_tf = anndata.read_h5ad(ad_tf_file)
    ad_rna = anndata.read_h5ad(ad_rna_file)
    cts = sorted(ad_rna.obs['celltype.calico'].unique())
    
    # read model
    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(model_file)
    
    # model params
    new_model = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=model.layers[-4].output,
    )
    w = model.layers[-3].get_weights()[0]
    
    # extract test sequence
    records = list(SeqIO.parse(fasta_file, "fasta"))
    seqs = [str(i.seq) for i in records]
    
    peak_gene_corr = pd.read_csv(corr_file, index_col=0)
    fimo = pd.read_csv(fimo_file, sep='\t', comment='#')
    
    # TF mRNA
    tf_rna = np.array(np.log2(ad_rna[:,tf_name].layers['denoised']/100+1)).flatten()
    tf_rna = pd.Series(tf_rna, index=ad_rna.obs_names)
    
    # filter by peak-gene correlation
    tmp = peak_gene_corr.loc[:,~peak_gene_corr.columns.isin(['Others','gene','peak'])]
    peak_gene_tmp = peak_gene_corr.loc[(tmp>peak_gene_thres).sum(axis=1)>0,:]
    
    tf_fimo = fimo.loc[(fimo['motif_id']==motif_id) & (fimo['sequence_name'].isin(peak_gene_tmp['peak'])),:]
        
    # remove duplicate entries
    tf_fimo = tf_fimo.sort_values(['sequence_name','start', 'strand']).groupby(['sequence_name','start']).first()
    tf_fimo = pd.DataFrame(tf_fimo.to_records())
    print('total fimo hits: %d'%tf_fimo.shape[0])

    # initialize
    ism_by_cell_m = np.empty((tf_fimo.shape[0], ad.shape[0]))
    target_genes_list = [] # targets genes 
    ism_pwm_cor_m = pd.DataFrame(np.nan, index=np.arange(tf_fimo.shape[0]), columns=cts) # correlation b/w ism and pwm
    ism_tfact_cor_m = pd.DataFrame(np.nan, index=np.arange(tf_fimo.shape[0]), columns=cts) # correlation b/w ism and tf activity
    ism_tfexp_cor_m = pd.DataFrame(np.nan, index=np.arange(tf_fimo.shape[0]), columns=cts) # correlation b/w ism and tf expression
    ism_peak_cor_m = pd.DataFrame(np.nan, index=np.arange(tf_fimo.shape[0]), columns=cts) # correlation b/ ism and peak
    
    for i in range(tf_fimo.shape[0]):
        
        peak_id = tf_fimo.iloc[i,:]['sequence_name']
        peak_index = int(peak_id.replace('peak_',''))-1
        start = tf_fimo.iloc[i,:]['start']-1
        end = tf_fimo.iloc[i,:]['stop']
        strand = tf_fimo.iloc[i,:]['strand']
        fimo_match = tf_fimo.iloc[i,:]['matched_sequence']
    
        pwm = pd.read_csv('%s/%s.csv'%(pwm_path, motif_id), index_col=0)
        pwm_rc = pwm.iloc[::-1,::-1]
        pwm_rc.index = ['A','C','G','T']
        if strand=='-': pwm = pwm_rc
    
        # check if fimo_match matches sequence
        if strand=='+':
            if fimo_match!=seqs[peak_index][start:end]: sys.exit("sequence doesn't match fimo.")
        else:
            fimo_match_rc = str(Seq(fimo_match).reverse_complement())
            if fimo_match_rc!=seqs[peak_index][start:end]: sys.exit("sequence doesn't match fimo.")
            
        acc, isms = motif_ism(new_model, w, seqs[peak_index], start, end)
        acc = pd.Series(acc, index=ad.obs_names)
        isms_norm = isms - np.repeat(isms.mean(axis=2)[:,:,np.newaxis], 4, axis=2)
            
        # per cell ism score: a single value per cell (isms_norm on ref nucleotide, average over the motif)
        dna_m = dna_1hot(seqs[peak_index][start:end])
        dna_m_broadcast = np.repeat(dna_m[np.newaxis,:,:], len(isms_norm), axis=0)
        cell_ism = (dna_m_broadcast * isms_norm).sum(axis=2).mean(axis=1)
        cell_ism = pd.Series(cell_ism, index=ad.obs_names)
        ism_by_cell_m[i,:] = cell_ism.values
    
        for ct in cts:
            
            choose_cells = ad_rna.obs_names[(ad_rna.obs['celltype.calico']==ct)].intersection(ad.obs_names)
            tf_activity = np.array(ad_tf[choose_cells,tf_name].X.flatten())
            ism_scores = cell_ism.loc[choose_cells]
            tmp_ism = isms_norm[ad.obs_names.isin(choose_cells)].mean(axis=0)
            
            ism_pwm_cor_m.loc[i, ct] = pearsonr(pwm.transpose().values.flatten(), tmp_ism.flatten())[0] # ism pwm cor
            ism_tfact_cor_m.loc[i, ct] = pearsonr(tf_activity, ism_scores)[0]  # ism tf-activity cor
            ism_tfexp_cor_m.loc[i, ct] = pearsonr(tf_rna.loc[choose_cells], ism_scores)[0] # ism tf-exp cor
            ism_peak_cor_m.loc[i, ct] = pearsonr(acc.loc[choose_cells], ism_scores)[0] # ism peak cor
    
        genes = peak_gene_tmp.loc[peak_gene_tmp['peak']==peak_id,'gene'].unique()
        target_genes = ' '.join(genes)
        target_genes_list += [target_genes]
        
        if (i%100==0): print(i)

    tf_fimo['gene'] = target_genes_list
    tf_fimo.to_csv('%s/fimo_hits.csv'%outdir)
    np.save('%s/ism_by_cell.npy'%outdir, ism_by_cell_m)
    
    ism_pwm_cor_m.to_csv('%s/ism_pwm_cor_m.csv'%outdir)
    ism_tfact_cor_m.to_csv('%s/ism_tfact_cor_m.csv'%outdir)
    ism_tfexp_cor_m.to_csv('%s/ism_tfexp_cor_m.csv'%outdir)
    ism_peak_cor_m.to_csv('%s/ism_peak_cor_m.csv'%outdir)
    
if __name__ == '__main__':
    main()