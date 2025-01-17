#!/bin/bash

# running in scbasset env

GENE=${1}
dist_thres=${2}
SCB_PATH='/home/yuanh/analysis/sc_basset/scbasset_package/scBasset/bin/multiome'


# files
peak_atlas='/home/yuanh/analysis/kidney_kenwhite/all_data_10.9.23/combined/atac/scbasset/processed/peaks.bed'
model_file='/home/yuanh/analysis/kidney_kenwhite/all_data_10.9.23/combined/atac/scbasset/output/best_model.h5'
ad_atac_file='/home/yuanh/analysis/kidney_kenwhite/all_data_10.9.23/combined/GRN_upstream/ad_atac_annot.h5ad'
ad_rna_file='/home/yuanh/analysis/kidney_kenwhite/all_data_10.9.23/combined/rna/anndata/ad_scvi.h5ad'
fasta_file='/home/yuanh/programs/genomes/mm10/mm10.fa'
meme_path='/scratch4/yuanh/meme_motifs/motif_databases_3_20_2022/CIS-BP_2.00/Mus_musculus.meme'
motif2tf='/scratch4/yuanh/meme_motifs/motif2tf_musmusculus.csv'
pwm_path='/home/yuanh/analysis/kidney_kenwhite/all_data_10.9.23/combined/atac/scbasset/analysis/motif_seqs/motif_pwms'
mkdir ${GENE}
cd ${GENE}

############################
# identify candidate peaks #
############################
# note: only works with multiome data

# 1. find peaks in +/-200kb around gene TSS, find genes in +/-400kb around gene TSS.
Rscript ${SCB_PATH}/peak_gene_nearby.R ${GENE} ${peak_atlas}

# 2. compute peak-gene-cor matrix. Identify candidate peaks.
${SCB_PATH}/scbasset_peak_gene_cor.py --ad_rna ${ad_rna_file} \
                                      --ad_atac ${ad_atac_file} \
                                      --model ${model_file} \
                                      --fasta ${fasta_file} \
                                      --gene ${GENE}  \
                                      --dist_thres ${dist_thres} \
                                      --thres 0.4 # peak-gene-cor to be considered linkage

##############
# ism + fimo #
##############
# provide candidate peaks (e.g. 20kb around TSS).
# perform ISM and FIMO. Filter FIMO hits based on ISM-PWM dot product and ISM score.
# note: works without RNA.
${SCB_PATH}/scbasset_peak_ism.py --anndata ${ad_atac_file} \
                                 --model ${model_file} \
                                 --fasta ${fasta_file} \
                                 --meme ${meme_path} \
                                 --bed peaks_filter/peaks.bed \
                                 --pwm ${pwm_path} \
                                 --groupby celltype \
                                 --out out \
                                 --fimo_thres 1e-3 \
                                 --dot_thres 0.01 \
                                 --ism_thres 0.05

#######################
# incoporate RNA info #
#######################
# note: only works with multiome data
#out/motif_hits_rna.csv: all signif motifs hits (by ISM and PWM) with RNA information.
#out/motif_hits_rna_filter.csv: all signif motif hits further filter by cor with RNA.
${SCB_PATH}/motif_filter.py --ad_rna ${ad_rna_file} \
                            --ad_atac ${ad_atac_file} \
                            --motif2tf ${motif2tf} \
                            --gene ${GENE} \
                            --groupby celltype \
                            --ism_tfrna_thres 0.2 \
                            --ism_targetrna_thres 0.2
