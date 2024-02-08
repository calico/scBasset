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

'''
scbasset_process.py
This function takes as input:
    - the preprocessed anndata (from scbasset_preprocess.py);
    - the scbasset trained model (from scbasset_train.py);
    - genome fasta file.
    - motif meme file.
    - gene name.

and outputs:
    - peaks around the gene (annotation, bed, fasta).
    - ism score for each peak.
    - desnoised accessibility scores at each peak.
    - fimo output for each peak.

'''

def main():
    usage = 'usage: %prog [options] <anndata file> <trained model>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='fasta',
                      help='fasta file. required.')
    parser.add_option('-m', dest='meme',
                      help='meme file. required.')
    parser.add_option('-g', dest='gene', 
                      help='gene name. required')
    
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error('must provide 2 arguments as inputs.')
    else:
        ad_file = args[0]
        trained_model = args[1]

    # parse options
    fasta_file = options.fasta
    meme_file = options.meme
    gene_name = options.gene
    
    outdir = gene_name
    os.makedirs(outdir, exist_ok=True)

    ad = anndata.read_h5ad(ad_file)
    if gene_name not in ad.var['gene_name'].values:
        sys.exit('gene name not found.')
    
    # write peak annotations
    peaks = ad.var.loc[ad.var['gene_name']==gene_name,:]
    peaks_gr = peaks.loc[:,['chr','start','end']]
    del peaks['chr']
    del peaks['start']
    del peaks['end']
    peak_names = peaks.index.values
    peaks.to_csv('%s/peaks_annot.csv'%outdir)
    
    # write bed file
    seqs_dna, seqs_coords = make_bed_seqs_from_df(peaks_gr, fasta_file, seq_len=1344, stranded=False)
    peaks_1344 = pd.DataFrame(seqs_coords)
    peaks_1344.columns = ['chr','start','end']
    peaks_1344.to_csv('%s/peaks.bed'%gene_name, index=False, header=False, sep='\t')
    
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
    
    print('start running ism.')
    
    # perform ism
    f1 = h5py.File("%s/ism_scores.h5"%outdir, "w")
    f2 = h5py.File("%s/ism_scores_norm.h5"%outdir, "w")
    for i in range(len(seqs_dna)):
        # ism scores
        seq_ref_1hot = dna_1hot(seqs_dna[i])
        m = ism(seq_ref_1hot, model)
        f1.create_dataset(peak_names[i], data=m)
        
        # normalize ism scores
        m_norm = m - np.repeat(m.mean(axis=2)[:,:,np.newaxis], 4, axis=2) # normalize on the 4 nucleotides 
        isms_ref = np.array([(t*seq_ref_1hot).sum(axis=1) for t in m_norm]) # only take score of the ref nucleotide 
        f2.create_dataset(peak_names[i], data=isms_ref)

    f1.close()
    f2.close()
    
    print('ism finished. start fimo.')

    # running fimo at a relaxed threshold
    cmd = 'fimo --thresh 1e-3 --oc %s/fimo_out %s %s/peaks.fasta'%(outdir, meme_file, outdir)
    subprocess.call(cmd, shell=True)
    
if __name__ == '__main__':
    main()