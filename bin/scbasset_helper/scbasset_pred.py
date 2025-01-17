#!/usr/bin/env python

import os
import anndata
import numpy as np
import pandas as pd
from scbasset.utils import *
import configargparse

def make_parser():
    parser = configargparse.ArgParser(
        description="scbasset predicts accessibility on given peaks.")
    parser.add_argument('--anndata', type=str,
                       help='anndata used to train scbasset.')
    parser.add_argument('--trained_model', type=str,
                       help='trained scbasset model.')
    parser.add_argument('--bed_file', type=str,
                       help='peak bed file to make prediction for.')
    parser.add_argument('--fasta_file', type=str,
                       help='genomic fasta file.')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='batch size.')
    parser.add_argument('--out_h5', type=str, default='denoise_output.h5',
                       help='output h5 file.')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    ad_file = args.anndata
    trained_model = args.trained_model
    bed_file = args.bed_file
    fasta_file = args.fasta_file
    batch_size = args.batch_size
    h5_file = args.out_h5

    ad = anndata.read_h5ad(ad_file)
    peaks = pd.read_csv(bed_file, sep='\t', header=None, index_col=False)
    n_pks = peaks.shape[0]
    n_cells = ad.shape[0]
    
    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model)

    #############
    # create h5 #
    #############
    if os.path.isfile(h5_file):
        os.remove(h5_file)
    
    h5f = h5py.File(h5_file, "w")
    h5f.create_dataset("cell_id", data=np.array(ad.obs_names, dtype="S"))
    h5f.create_dataset("pred", dtype="float32", shape=(n_pks, n_cells))
    
    ######################
    # predict in batches #
    ######################
    n_batches = int(np.ceil(peaks.shape[0] / batch_size))
    batches = np.array_split(np.arange(peaks.shape[0]), n_batches)
    
    for idxs in batches:
        tmp_pks = peaks.iloc[idxs,:]
        seqs_dna, seqs_coords = make_bed_seqs_from_df(tmp_pks, fasta_file, seq_len=1344, stranded=False)
        seqs_1hot = np.array([dna_1hot(i) for i in seqs_dna])
        preds = imputation_Y_normalize(seqs_1hot, model, scale_method='sigmoid')
        h5f['pred'][idxs,:] = preds
        print('%d pks predicted.' %idxs[-1])
    
    h5f.close()

if __name__ == "__main__":
    main()
