#/bin/bash

python ../../bin/scbasset_preprocess.py --ad_file atac_ad.h5ad --input_fasta /home/yuanh/genomes/hg38/hg38.fa
python ../../bin/scbasset_train.py --input_folder processed/ --epochs 1000
