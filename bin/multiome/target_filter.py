#!/usr/bin/env python
import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import pickle
from optparse import OptionParser

def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('--ad_atac', dest='ad_atac', help='ATAC anndata.')
    parser.add_option('--corr_file', dest='corr_file', help='peak-gene corr file.')
    parser.add_option('--ism_folder', dest='ism_folder', default='ism_out', help='TF anndata.')
    parser.add_option('--ism_pwm_cor',  dest='t1', type="float", default=0.6, help='ism pwm correlation threshold.')
    parser.add_option('--ism_peak_cor',  dest='t2', type="float", default=0.4, help='ism peak correlation threshold.')
    parser.add_option('--ism_tfact_cor',  dest='t3', type="float", default=0.4, help='ism TF activity correlation threshold.')
    parser.add_option('--ism_abs',  dest='t4', type="float", default=0.1, help='absolute ism threshold.')
    parser.add_option('--outdir', dest='outdir', default='./', help='output directory.')
    
    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()
    ad_file = options.ad_atac
    ism_folder = options.ism_folder
    corr_file = options.corr_file
    outdir = options.outdir
    t1 = options.t1
    t2 = options.t2
    t3 = options.t3
    t4 = options.t4
    
    ad = anndata.read_h5ad(ad_file)
    
    tf_fimo = pd.read_csv('%s/fimo_hits.csv'%ism_folder, index_col=0)
    ism_pwm_cor_m = pd.read_csv('%s/ism_pwm_cor_m.csv'%ism_folder, index_col=0)
    ism_peak_cor_m = pd.read_csv('%s/ism_peak_cor_m.csv'%ism_folder, index_col=0)
    ism_tfact_cor_m = pd.read_csv('%s/ism_tfact_cor_m.csv'%ism_folder, index_col=0)
    ism_tfexp_cor_m = pd.read_csv('%s/ism_tfexp_cor_m.csv'%ism_folder, index_col=0)
    ism_by_cell_m = np.load('%s/ism_by_cell.npy'%ism_folder)
    
    peak_gene_corr = pd.read_csv(corr_file, index_col=0)
    tmp = peak_gene_corr.loc[:,~peak_gene_corr.columns.isin(['Others','gene','peak'])]
    peak_gene_tmp = peak_gene_corr.loc[(tmp>0.6).sum(axis=1)>0,:]
    
    # add target gene info
    target_genes_list = []
    for i in range(tf_fimo.shape[0]):
        peak_id = tf_fimo.iloc[i,:]['sequence_name']
        peak_index = int(peak_id.replace('peak_',''))-1
        genes = peak_gene_tmp.loc[peak_gene_tmp['peak']==peak_id,'gene'].unique()
        target_genes = ' '.join(genes)
        target_genes_list += [target_genes]
    tf_fimo['gene'] = target_genes_list
    
    # ism by cell matrix
    ism_m = pd.DataFrame(ism_by_cell_m).transpose()
    ism_m['celltype'] = ad.obs['celltype'].values
    ism_m = ism_m.groupby('celltype').mean()
    ism_m = ism_m.loc[ism_pwm_cor_m.columns,:].transpose()
    
    # 5 filters applied to select motif hits:
    filter_hits = tf_fimo.loc[((ism_pwm_cor_m>t1) & 
                               (ism_peak_cor_m>t2) & 
                               (ism_tfact_cor_m>t3) & 
                               ism_m.abs()>t4).sum(axis=1)>0,:]
    
    filter_hits.to_csv('%s/fimo_hits_filter.csv'%ism_folder)
    
    targets = sum(filter_hits['gene'].str.split(' ').tolist(), [])
    targets = list(set(targets))
    gs = {}
    gs['targets'] = targets
    
    with open('%s/gs.pkl'%outdir, 'wb') as f:
        pickle.dump(gs, f)
    
    pd.DataFrame(gs).to_csv('%s/targets.csv'%outdir)

if __name__ == '__main__':
    main()