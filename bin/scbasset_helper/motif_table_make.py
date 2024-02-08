import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from scbasset.utils import *
from optparse import OptionParser

'''
motif_table_make.py
This function takes as input:
    - the preprocessed anndata (from scbasset_preprocess.py);
    - folder for all pwm.csv.
    - folder for gene analysis.
and outputs:
    - denoised_acc.pdf. accessibility for the peaks of interest.
    - motif_table_all.csv. A table of all motifs, and ISM scores for each cell group.

'''

# this code is only suitable for analyze KenWhite AD vs. CAS.

def main():
    usage = 'usage: %prog [options] <anndata file> <pwm folder> <gene folder>'
    parser = OptionParser(usage)    
    (options, args) = parser.parse_args()
    if len(args) != 3:
        parser.error('must provide 3 arguments as inputs.')
    else:
        ad_file = args[0]
        pwm_path = args[1]
        gene_dir = args[2]

    # parse options    
    ad = anndata.read_h5ad(ad_file)

    # define cell groups
    ad.obs['celltype'] = ad.obs['celltype_sex1'].str.replace(' .*','',regex=True)
    ad.obs['treatment'] = ad.obs['treatment'].cat.reorder_categories(['CAS', 'AD'])
    ad.obs['group'] = ad.obs['celltype'].str.cat(ad.obs['treatment'], '_')
    groups = sorted(ad.obs['group'].unique())
    
    # plot accessibility change
    peaks = pd.read_csv('%s/peaks_annot.csv'%gene_dir, index_col=0)
    acc = pd.read_csv('%s/desnoised_acc.csv'%gene_dir, index_col=0)
    for i in acc.columns:
        ad.obs[i] = acc[i].values
    
    cts = sorted(ad.obs['celltype'].unique())
    f, axs = plt.subplots(ncols=1, nrows=acc.shape[1], figsize=(8, acc.shape[1]*3))
    for i in range(acc.shape[1]):
        sns.boxplot(x='celltype', y=acc.columns[i], hue='treatment', data=ad.obs, ax=axs[i], order=cts)
        axs[i].set_title(acc.columns[i])
        axs[i].set_ylim(0, 1)
        axs[i].set_ylabel('denoised accessibility')
        axs[i].set_xlabel('')
        axs[i].axhline(y=0.5, color='grey', linestyle='--')
        axs[i].set_xticklabels(axs[i].get_xticklabels(),rotation=45)
        axs[i].legend([],[], frameon=False)
    f.tight_layout()
    f.savefig('%s/denoised_acc.pdf'%gene_dir)
    
    # 1-hot-encoded inputs
    seq_ref_1hots = {}
    fasta_open = pysam.Fastafile('%s/peaks.fasta'%gene_dir)
    for peak_name in peaks.index:
        seq_i = fasta_open.fetch(peak_name)
        seq_ref_1hots[peak_name] = dna_1hot(seq_i)
    fasta_open.close()

    ### summarize per cell ism scores to per group
    scores = {}
    f = h5py.File("%s/ism_scores.h5"%gene_dir, "r")
    for peak_name in peaks.index:
        a = f[peak_name][:]
        # normalize on the 4 nucleotides
        a_norm = a - np.repeat(a.mean(axis=2)[:,:,np.newaxis], 4, axis=2)
        # aggregate by group
        a_out = []
        for t in groups:
            a_out += [a_norm[(ad.obs['group']==t).values,:,:].mean(axis=0)]
        a_norm = np.array(a_out)
        # save
        scores[peak_name] = a_norm
    f.close()
    
    ### add ism scores to the motif tracks
    hits = pd.read_csv('%s/fimo_out/fimo.tsv'%gene_dir, sep='\t', comment='#')
    hits['hit_id'] = [i[0] + '_' + i[1] + '_' + str(i[2]) + '_' + i[3] for i in zip(hits['sequence_name'], hits['motif_id'], hits['start'], hits['strand'])]
    hits.index = hits['hit_id']
    hits['cor_max'] = 0
    hits['ism_over_bg_max'] = 0

    motif_ism_m = []
    for hit_name in hits.index:
        motif_name = hits.loc[hit_name, 'motif_id']
        pwm = pd.read_csv('%s/%s.csv'%(pwm_path, motif_name), index_col=0)
        start_site = int(hits.loc[hit_name, 'start']-1)
        peak_id = hits.loc[hit_name, 'sequence_name']
        
        ism_matrix = scores[peak_id][:,start_site:(start_site+pwm.shape[1]),:]
        pwm = pwm.iloc[:,:ism_matrix.shape[1]] # in case pwm is longer than ism_matrix
    
        # correlation b/w ism and pwm for each group
        cors = [pearsonr(pwm.transpose().values.flatten(), t.flatten())[0] for t in ism_matrix]
    
        # isms: ism scores
        isms_ref = [(t*seq_ref_1hots[peak_id]).sum(axis=1) for t in scores[peak_id]]
        isms = np.array([t[start_site:(start_site+pwm.shape[1])].mean() for t in isms_ref])

        # ism_over_bg: ism vs. background (log2fc)
        a = np.abs(np.array([t[start_site:(start_site+pwm.shape[1])].mean() for t in isms_ref]))
        b = np.abs(np.array([t.mean() for t in isms_ref]))
        isms_over_bg = np.log2(a/b)
    
        # save
        hits.loc[hit_name, 'cor_max'] = np.max(cors)
        hits.loc[hit_name, 'ism_over_bg_max'] = np.max(isms_over_bg)
        motif_ism_m += [isms]
        
    motif_ism_df = pd.DataFrame(motif_ism_m, columns=groups, index=hits.index)
    hits_output = pd.concat([hits, motif_ism_df], axis=1)
    hits_output.to_csv('%s/motif_table_all.csv'%gene_dir)
    
if __name__ == '__main__':
    main()