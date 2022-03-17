cp ~/sc_basset/datasets/10x_ARC_PBMC/raw/pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz ./
gunzip pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz
sort -k4,4 pbmc_granulocyte_sorted_3k_atac_fragments.tsv.gz > pbmc_granulocyte_sorted_3k_atac_fragments.bed
gzip pbmc_granulocyte_sorted_3k_atac_fragments.bed
snaptools snap-pre  \
	--input-file=pbmc_granulocyte_sorted_3k_atac_fragments.bed.gz  \
	--output-snap=pbmc_granulocyte_sorted_3k_atac_fragments.snap  \
	--genome-name=hg38  \
	--genome-size=/home/yuanh/genomes/hg38/hg38.chrom.sizes  \
	--min-mapq=30  \
	--min-flen=50  \
	--max-flen=1000  \
	--keep-chrm=TRUE  \
	--keep-single=FALSE  \
	--keep-secondary=False  \
	--overwrite=True  \
	--max-num=20000 \
	--min-cov=5  \
	--verbose=True

snaptools snap-add-bmat	\
	--snap-file=pbmc_granulocyte_sorted_3k_atac_fragments.snap	\
	--bin-size-list 1000 5000 10000	\
	--verbose=True
