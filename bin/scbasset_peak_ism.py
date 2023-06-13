#!/usr/bin/env python

import h5py
import os
import glob
import sys
import subprocess

import numpy as np
import scanpy as sc
import pandas as pd
from optparse import OptionParser
from scbasset.utils import *
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_ind

'''
scbasset_peak_ism.py
# this function requires fimo installed.
# this function perform motif-ism analysis for peaks around gene-of-interest.

This function takes as input:
    - the preprocessed anndata (from scbasset_preprocess.py);
    - the scbasset trained model (from scbasset_train.py);
    - genome fasta file.
    - motif meme file.
    - bed file for a set of peaks associated with a gene.

and outputs:
    - ism scores for each peak.
    - desnoised accessibility scores at each peak.
    - fimo output for each peak.

'''

def main():
    usage = 'usage: %prog [options] <anndata file> <trained model>'
    parser = OptionParser(usage)
    parser.add_option('--anndata', dest='ad', help='anndata.')
    parser.add_option('--model', dest='model', help='trained scBasset model h5.')
    parser.add_option('--fasta', dest='fasta', help='fasta file.')
    parser.add_option('--meme', dest='meme', help='meme file.')
    parser.add_option('--bed', dest='bed', help='named bed file, 4th column contains peak names.')
    parser.add_option('--pwm', dest='pwm_path', help='pwm folder path. Output of process_meme.R.')
    parser.add_option('--groupby', dest='groupby', help='column in ad.obs to group cells. usually cell type or cluster.')
    parser.add_option('--out', dest='out', default='output', help='output directory, optional.')
    parser.add_option('--fimo_thres',  dest='fimo_thres', type="float", default=1e-4, help='fimo pval threshold.')
    parser.add_option('--dot_thres',  dest='dot_thres', type="float", default=0.01, help='ism-pwm dot product qval threshold.')
    parser.add_option('--ism_thres',  dest='ism_thres', type="float", default=0.05, help='ism qval threshold.')
    
    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()
    
    # parse options
    ad_file = options.ad
    trained_model = options.model
    fasta_file = options.fasta
    meme_file = options.meme
    bed_file = options.bed
    pwm_path = options.pwm_path
    groupby = options.groupby
    outdir = options.out
    fimo_thres = options.fimo_thres
    dot_thres = options.dot_thres
    ism_thres = options.ism_thres

    bed = pd.read_csv(bed_file, sep='\t', header=None)
    peak_names = bed.iloc[:,3]
    os.makedirs(outdir, exist_ok=True)
    
    ad = anndata.read_h5ad(ad_file)
    groups = sorted(ad.obs[groupby].unique())
    
    ##############
    # save peaks #
    ##############
    
    # write bed file
    seqs_dna, seqs_coords = make_bed_seqs_from_df(bed, fasta_file, seq_len=1344, stranded=False)
    peaks_1344 = pd.DataFrame(seqs_coords)
    peaks_1344.columns = ['chr','start','end']
    peaks_1344.to_csv('%s/peaks.bed'%outdir, index=False, header=False, sep='\t')
    
    # write fasta file
    fa = open("%s/peaks.fasta"%outdir, "w")
    for i in range(len(seqs_dna)):
        fa.write(">" + peak_names[i] + "\n" +seqs_dna[i] + "\n")
    fa.close()
    
    # denoised accessibility per cell
    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model)
    y = pred_on_fasta('%s/peaks.fasta'%outdir, model, scale_method='sigmoid')
    y = pd.DataFrame(y, index=peak_names, columns=ad.obs_names).transpose()
    y.to_csv('%s/desnoised_acc.csv'%outdir)
    
    ###############
    # compute ISM #
    ###############
    print('start running ism.')
    ism_ref_dict = {}

    # perform ism
    f1 = h5py.File("%s/ism_scores.h5"%outdir, "w")
    f2 = h5py.File("%s/ism_ref.h5"%outdir, "w")
    for i in range(len(seqs_dna)):
        # ism scores
        seq_ref_1hot = dna_1hot(seqs_dna[i])
        m = ism(seq_ref_1hot, model)
        f1.create_dataset(peak_names[i], data=m)
        
        # normalize ism scores
        isms_ref = np.array([(t*seq_ref_1hot).sum(axis=1) for t in m]) # only take score of the ref nucleotide 
        ism_ref_dict[peak_names[i]] = isms_ref
        f2.create_dataset(peak_names[i], data=isms_ref)

    f1.close()
    f2.close()
    
    print('ism finished. start fimo.')

    # running fimo at a relaxed threshold
    cmd = 'fimo --thresh %.e --oc %s/fimo_out %s %s/peaks.fasta'%(fimo_thres, outdir, meme_file, outdir)
    subprocess.call(cmd, shell=True)
    
    ######################
    # combine ISM + FIMO #
    ######################
    
    print('FIMO finished. computing FIMO motif ISM scores.')
    
    ### group*1344*4 tensor
    group_ism = {}
    f = h5py.File("%s/ism_scores.h5"%outdir, "r")
    for peak_name in peak_names:
        a = f[peak_name][:]
        a_out = []
        for t in groups:
            a_out += [a[(ad.obs[groupby]==t).values,:,:].mean(axis=0)]
        a = np.array(a_out)
        # save
        group_ism[peak_name] = a
    f.close()

    ### group*1344 reference ism matrix
    group_ism_ref = {}
    for peak_name in peak_names:
        a = ism_ref_dict[peak_name]
        a_out = []
        for t in groups:
            a_out += [a[(ad.obs[groupby]==t).values,:].mean(axis=0)]
        a = np.array(a_out)
        # save
        group_ism_ref[peak_name] = a
    
    # shuffle ism matrix for null dot product distribution
    shuffle_ism = {}
    for peak_id in group_ism:
        tmp = group_ism[peak_id].mean(axis=0)
        np.random.shuffle(tmp)
        shuffle_ism[peak_id] = tmp
    
    # combine with FIMO
    hits = pd.read_csv('%s/fimo_out/fimo.tsv'%outdir, sep='\t', comment='#')
    hits['hit_id'] = [i[0]+'_'+i[1]+'_'+str(i[2])+'_'+i[3] for i in zip(hits['sequence_name'],hits['motif_id'], 
                                                                        hits['start'], hits['strand'])]
    hits.index = hits['hit_id']
    hits['dot'] = 0
    hits['dot_pval'] = 1
    hits['ism'] = 0
    hits['ism_pval'] = 1

    m_ism = np.zeros((hits.shape[0], ad.shape[0]))
    
    counter = 0
    for hit_name in hits.index:
        motif_name = hits.loc[hit_name, 'motif_id']
        
        # determine pwm
        # account for reverse complement
        strand = hits.loc[hit_name, 'strand']
        pwm = pd.read_csv('%s/%s.csv'%(pwm_path, motif_name), index_col=0)
        pwm_rc = pwm.iloc[::-1,::-1]
        pwm_rc.index = ['A','C','G','T']
        if strand=='-': pwm = pwm_rc
        
        start_site = int(hits.loc[hit_name, 'start']-1)
        peak_id = hits.loc[hit_name, 'sequence_name']
        
        # ism subset
        ism_matrix = group_ism[peak_id][:,start_site:(start_site+pwm.shape[1]),:]
        
        # pwm
        pwm = pwm.iloc[:,:ism_matrix.shape[1]] # in case pwm is longer than ism_matrix
        pwm = pwm.transpose().values
    
        # test pwm-ism dot product significance: permutation
        dot_tmp = [np.dot(pwm.flatten(), t.flatten()) for t in ism_matrix]
        max_dot = np.max(dot_tmp)
        max_idx = np.argmax(dot_tmp)
        # dot product empirical p-value
        tmp = shuffle_ism[peak_id]
        bg_dots = []
        for i in range(tmp.shape[0]-pwm.shape[0]):    
            a = tmp[i:(i+pwm.shape[0]),:]
            bg_dots += [np.dot(pwm.flatten(), a.flatten())]
        p = (np.array(bg_dots)>max_dot).sum()/len(bg_dots)
            
        hits.loc[hit_name, 'dot'] = max_dot
        hits.loc[hit_name, 'dot_pval'] = p

        # test ism significance: t-test
        ref = group_ism_ref[peak_id][max_idx]
        ref_pwm = ref[start_site:(start_site+pwm.shape[0])]
        ref_other = np.delete(ref, np.s_[start_site:(start_site+pwm.shape[0])])        
        hits.loc[hit_name, 'ism'] = np.float64(ref_pwm.mean() - ref_other.mean()) # cast to float64 because my np.fl32 has missing attributes
        hits.loc[hit_name, 'ism_pval'] = ttest_ind(ref_pwm, ref_other, 
                                                   equal_var=False).pvalue

        # add to m_ism
        m_ism[counter,:] = ism_ref_dict[peak_id][:,start_site:(start_site+pwm.shape[0])].mean(axis=1)
        counter += 1
        
    hits['dot_qval'] = fdrcorrection(hits['dot_pval'])[1]
    hits['ism_qval'] = fdrcorrection(hits['ism_pval'])[1]
    
    m_ism = m_ism[(hits['dot_qval']<dot_thres) & (hits['ism_qval']<ism_thres),:]
    hits = hits.loc[(hits['dot_qval']<dot_thres) & (hits['ism_qval']<ism_thres),:]
    
    np.save('%s/ism_matrix.npy'%outdir, m_ism)
    hits.to_csv('%s/motif_hits.csv'%outdir)
    
if __name__ == '__main__':
    main()