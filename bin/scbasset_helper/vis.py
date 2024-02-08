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
    usage = 'usage: %prog [options] <peak_id> <pwm_start> <pwm_len> <group_info> <cts> <out_file> '
    parser = OptionParser(usage)
    parser.add_option('--vmin', type='float', dest='vmin', default=-0.5,
                      help='fasta file. required.')
    parser.add_option('--vmax', type='float', dest='vmax', default=0.5,
                      help='meme file. required.')
    
    (options, args) = parser.parse_args()
    peak_id = args[0]
    pwm_start = int(args[1])
    pwm_len = int(args[2])
    group_file = args[3] # csv file indicating the groups
    cts_str = args[4] # a string, groups are separated by ':'
    out_file = args[5]

    # parse options
    vmin = options.vmin
    vmax = options.vmax
    
    ad_atac_file = '/home/yuanh/analysis/kidney_kenwhite/CKD_aug_10/scbasset_atac/analysis2/results/ad_scbasset.h5ad'
    fasta_file = '/home/yuanh/analysis/kidney_kenwhite/CKD_aug_10/ism_methods/run_fimo/peaks.fasta'
    model_file = '/home/yuanh/analysis/kidney_kenwhite/CKD_aug_10/scbasset_atac/output/best_model.h5'

    # define cell groups
    ad = anndata.read_h5ad(ad_atac_file)
    ad.obs['celltype'] = ad.obs['celltype_sex1'].str.replace(' .*','')
    ad.obs['treatment'] = ad.obs['treatment'].cat.reorder_categories(['CAS', 'AD'])
    ad.obs['group'] = pd.read_csv(group_file, index_col=0).iloc[:,0]
    cts = cts_str.split(':')

    model = make_model(32, ad.shape[0], show_summary=False)
    model.load_weights(model_file)

    # extract test sequence
    records = list(SeqIO.parse(fasta_file, "fasta"))
    seqs = [str(i.seq) for i in records]
    
    # peak index
    peak_index = int(peak_id.replace('peak_',''))-1
    
    # run ism
    seq_ref_1hot = dna_1hot(seqs[peak_index])
    m = ism(seq_ref_1hot, model)

    # plot ism in 100bp window around motif of interest
    a_norm = m - np.repeat(m.mean(axis=2)[:,:,np.newaxis], 4, axis=2)
    start = max(pwm_start-50, 0)
    pwm_pos = pwm_start - start
    end = min(start+100, 1344)
    f, axs = plt.subplots(nrows=len(cts), figsize=(20, len(cts)*2))
    for j in range(len(cts)):
        cells = np.where(ad.obs['group']==cts[j])[0]
        agg_profile = a_norm[cells,:,:].mean(axis=0)
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