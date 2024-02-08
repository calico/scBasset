#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import pandas as pd
import logomaker
from scipy.stats import pearsonr
from scbasset.utils import *
import matplotlib.patches as patches
from optparse import OptionParser
import subprocess
import pyBigWig

def plot_logo(m, ymin, ymax, ax, title, pwm_pos, pwm_len):
    nn_logo = logomaker.Logo(m, ax=ax, baseline_width=0)
    # style using Logo methods
    nn_logo.style_spines(visible=False)
    nn_logo.style_spines(spines=['left'], visible=True, bounds=[ymin, ymax])
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    
    # label the pwm
    rect = patches.Rectangle((pwm_pos-1.5, ymin+0.01), pwm_len, ymax-(ymin+0.01), linewidth=1, 
                             edgecolor='b', facecolor='none', linestyle='dashed')
    ax.add_patch(rect)

def main():
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('--anndata', dest='ad', help='anndata.')
    parser.add_option('--model', dest='model', help='trained scBasset model h5.')
    parser.add_option('--fasta', dest='fasta', help='fasta file.')
    parser.add_option('--bed', dest='bed', help='named bed file, 4th column contains peak names.')
    parser.add_option('--logo_path', dest='logo_path', help='path to motif logos.')
    parser.add_option('--table', dest='table', help='motif hit table.')
    parser.add_option('--groupby', dest='groupby', help='column in ad.obs to group cells. usually cell type or cluster.')
    parser.add_option('--cts', dest='cts', help='a string, groups to be plotted are separated by :')
    parser.add_option('--phylop', dest='phylop', help='phylop bigwig file.')
    parser.add_option('--outdir', dest='out', default='test', help='output directory.')
    parser.add_option('--vmax', type='float', dest='vmax', default=None, help='ymax.')

    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()
    
    # parse options
    ad_file = options.ad
    trained_model = options.model
    fasta_file = options.fasta
    bed_file = options.bed
    logo_path = options.logo_path
    table_file = options.table
    groupby = options.groupby
    cts_str = options.cts
    bw_file = options.phylop
    outdir = options.out
    vmax = options.vmax
    
    motifs = pd.read_csv(table_file, index_col=0)
    
    # define cell groups
    ad = anndata.read_h5ad(ad_file)
    cts = cts_str.split(':')

    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model)

    # extract test sequence
    bed = pd.read_csv(bed_file, sep='\t', header=None)
    peak_names = bed.iloc[:,3]
    seqs_dna, seqs_coords = make_bed_seqs_from_df(bed, fasta_file, seq_len=1344, stranded=False)
    seq_dict = {}
    seq_cor_dict = {}
    for i in range(len(peak_names)):
        seq_dict[peak_names[i]] = seqs_dna[i]
        seq_cor_dict[peak_names[i]] = seqs_coords[i]
    
    # read phylop track
    bw = pyBigWig.open(bw_file)
    
    os.makedirs(outdir, exist_ok=True)
    
    if vmax is None:
        vmax = motifs['ism'].abs().max()+0.05
        vmin = -vmax
    else:
        vmax = abs(vmax)
        vmin = -vmax
            
    for i in motifs.index[:5]:
        peak_id = motifs.loc[i, 'sequence_name']
        tf = motifs.loc[i, 'tf']
        motif_id = motifs.loc[i, 'motif_id']
        strand = motifs.loc[i, 'strand']
        pwm_start = motifs.loc[i, 'start']
        pwm_len = motifs.loc[i, 'stop'] - motifs.loc[i, 'start']
        
        # phylop track
        phylop = bw.values(seq_cor_dict[peak_id][0], seq_cor_dict[peak_id][1], seq_cor_dict[peak_id][2])        
        
        # run ism
        seq_ref_1hot = dna_1hot(seq_dict[peak_id])
        m = ism(seq_ref_1hot, model)

        # plot ism in 100bp window around motif of interest
        start = max(pwm_start-40, 0)
        pwm_pos = pwm_start - start
        end = min(start+80, 1344)
        f, axs = plt.subplots(nrows=len(cts)+1, figsize=(8, (len(cts)+1)))
        for j in range(len(cts)):
            cells = np.where(ad.obs[groupby]==cts[j])[0]
            agg_profile = m[cells,:,:].mean(axis=0)
            # plot
            toplot = pd.DataFrame(seq_ref_1hot * agg_profile, columns = ['A', 'C', 'G', 'T'])
            toplot = toplot.iloc[start:end,:]
            toplot.index = np.arange(toplot.shape[0])
            if j==0:
                plot_logo(toplot, vmin, vmax, axs[j], 'in silico saturation mutagenesis', pwm_pos, pwm_len)
            else:
                plot_logo(toplot, vmin, vmax, axs[j], '', pwm_pos, pwm_len)
            if j<2: axs[j].set_xticks([])
            axs[j].set_ylabel(cts[j])
        
        # plot phylop
        sns.lineplot(np.arange(end-start), phylop[start:end], ax=axs[j+1])
        axs[j+1].set_ylabel('phyloP')
        axs[j+1].set_xlim(0, end-start-1)
        
        f.tight_layout()
        f.savefig('%s/%s_%s_%d.png'%(outdir, peak_id, tf, pwm_start))
        
        if strand=='+':
            cmd = 'cp %s/%s.png %s/%s_%s_%d_motif.png' %(logo_path, motif_id, outdir, peak_id, tf, pwm_start)
        else:
            cmd = 'cp %s/%s_rc.png %s/%s_%s_%d_motif.png' %(logo_path, motif_id, outdir, peak_id, tf, pwm_start)
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()