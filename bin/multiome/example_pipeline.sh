#!/bin/bash

# running in tf28 (my scBasset env)

GENE=${1}
SCB_PATH='/home/yuanh/analysis/sc_basset/scbasset_package/scBasset/bin/multiome'

############################
# identify candidate peaks #
############################
# note: only works with multiome data
mkdir ${GENE}
cd ${GENE}

# 1. find peaks in +/-200kb around gene TSS, find genes in +/-400kb around gene TSS.
Rscript ${SCB_PATH}/peak_gene_nearby.R ${GENE}

# 2. compute peak-gene-cor matrix. Identify candidate peaks.
${SCB_PATH}/scbasset_peak_gene_cor.py --ad_rna /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scvi_rna/ad_scvi.h5ad \
                                      --ad_atac /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/Kl_regulation/ism_analysis/ad.h5ad \
                                      --model /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scbasset_atac/output/best_model.h5 \
                                      --fasta /home/yuanh/programs/genomes/mm10/mm10.fa \
                                      --gene ${GENE}  \
                                      --thres 0.4 # peak-gene-cor to be considered linkage

##############
# ism + fimo #
##############
# provide candidate peaks (e.g. 20kb around TSS).
# perform ISM and FIMO. Filter FIMO hits based on ISM-PWM dot product and ISM score.
# note: works without RNA.
${SCB_PATH}/scbasset_peak_ism.py --anndata /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/Kl_regulation/ism_analysis/ad.h5ad \
                                 --model /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scbasset_atac/output/best_model.h5 \
                                 --fasta /home/yuanh/programs/genomes/mm10/mm10.fa \
                                 --meme /scratch4/yuanh/meme_motifs/motif_databases_3_20_2022/CIS-BP_2.00/Mus_musculus.meme \
                                 --bed peaks_filter/peaks.bed \
                                 --pwm /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scbasset_atac/analysis2/motif_seqs/motif_pwms \
                                 --groupby group \
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
${SCB_PATH}/motif_filter.py --ad_rna /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scvi_rna/ad_scvi.h5ad \
                            --ad_atac /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/Kl_regulation/ism_analysis/ad.h5ad \
                            --motif2tf /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scbasset_atac/analysis2/motif2tf.csv \
                            --gene ${GENE} \
                            --ism_tfrna_thres 0.2 \
                            --ism_targetrna_thres 0.2

########
# plot #
########
# plot gene expression
# plot accessibility change
${SCB_PATH}/plot_change.py --ad_rna /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scvi_rna/ad_scvi.h5ad \
                           --ad_atac /home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/Kl_regulation/ism_analysis/ad.h5ad \
                           --gene ${GENE}

