cp ~/sc_basset/datasets/10x_ARC_mouse_brain/raw/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz ./
gunzip e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz
sort -k4,4 e18_mouse_brain_fresh_5k_atac_fragments.tsv > e18_mouse_brain_fresh_5k_atac_fragments.bed
gzip e18_mouse_brain_fresh_5k_atac_fragments.bed
snaptools snap-pre  \
	--input-file=e18_mouse_brain_fresh_5k_atac_fragments.bed.gz  \
	--output-snap=e18_mouse_brain_fresh_5k_atac_fragments.snap  \
	--genome-name=mm10  \
	--genome-size=/home/yuanh/genomes/mm10/mm10.chrom.sizes  \
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
	--snap-file=e18_mouse_brain_fresh_5k_atac_fragments.snap	\
	--bin-size-list 1000 5000 10000	\
	--verbose=True
