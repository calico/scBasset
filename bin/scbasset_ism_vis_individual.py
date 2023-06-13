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
    parser.add_option('--fasta', dest='fasta', help='peaks fasta file.')
    parser.add_option('--peakid', dest='peak_id', help='peak id of interest.')
    parser.add_option('--pwm_start', dest='pwm_start', type='int', help='start position of the motif.')
    parser.add_option('--pwm_len', dest='pwm_len', type='int', help='length of the pwm.')
    parser.add_option('--groupby', dest='groupby', help='column in ad.obs to group cells. usually cell type or cluster.')
    parser.add_option('--cts', dest='cts', help='a string, groups to be plotted are separated by :')
    parser.add_option('--outfile', dest='out', default='test.png', help='output file name, optional.')
    parser.add_option('--vmin', type='float', dest='vmin', default=-0.5, help='ymin.')
    parser.add_option('--vmax', type='float', dest='vmax', default=0.5, help='ymax.')

    ################
    # parse inputs #
    ################
    (options, args) = parser.parse_args()
    
    # parse options
    ad_file = options.ad
    trained_model = options.model
    fasta_file = options.fasta
    peak_id = options.peak_id
    pwm_start = options.pwm_start
    pwm_len = options.pwm_len
    groupby = options.groupby
    cts_str = options.cts
    out_file = options.out
    vmin = options.vmin
    vmax = options.vmax
    
    # define cell groups
    ad = anndata.read_h5ad(ad_file)
    cts = cts_str.split(':')

    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(trained_model)

    # extract test sequence
    seq_dict = {rec.id : rec.seq for rec in SeqIO.parse(fasta_file, "fasta")}
    
    # run ism
    seq_ref_1hot = dna_1hot(seq_dict[peak_id])
    m = ism(seq_ref_1hot, model)

    # plot ism in 100bp window around motif of interest
    start = max(pwm_start-40, 0)
    pwm_pos = pwm_start - start
    end = min(start+80, 1344)
    f, axs = plt.subplots(nrows=len(cts), figsize=(8, len(cts)*1))
    for j in range(len(cts)):
        cells = np.where(ad.obs[groupby]==cts[j])[0]
        agg_profile = m[cells,:,:].mean(axis=0)
        # plot
        toplot = pd.DataFrame(seq_ref_1hot * agg_profile, columns = ['A', 'C', 'G', 'T'])
        toplot = toplot.iloc[start:end,:]
        toplot.index = np.arange(toplot.shape[0])
        if j==0:
            plot_logo(toplot, vmin, vmax, axs[j], 'test', pwm_pos, pwm_len)
        else:
            plot_logo(toplot, vmin, vmax, axs[j], '', pwm_pos, pwm_len)
        if j<2: axs[j].set_xticks([])
        axs[j].set_ylabel(cts[j])
    f.tight_layout()
    
    f.savefig('%s'%out_file)

if __name__ == '__main__':
    main()