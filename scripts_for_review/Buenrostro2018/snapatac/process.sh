# for snapATAC
samtools view merged.sort.bam -H > merged.sort.header.sam

# create a bam file with the barcode embedded into the read name
cat <( cat merged.sort.header.sam ) \
<( samtools view merged.sort.bam | awk '{for (i=12; i<=NF; ++i) { if ($i ~ "^RG:Z:"){ td[substr($i,1,2)] = substr($i,6,length($i)-5); } }; printf "%s:%s\n", td["RG"], $0 }' ) \
| samtools view -bS - > merged.sort.snap.bam

samtools view merged.sort.snap.bam | head 
samtools sort -n -@ 10 -m 1G merged.sort.snap.bam -o merged.sort.snap.nsort.bam

snaptools snap-pre  \
	--input-file=merged.sort.snap.nsort.bam  \
	--output-snap=merged.sort.snap  \
	--genome-name=hg19  \
	--genome-size=/home/yuanh/genomes/hg19/hg19.chrom.sizes  \
	--min-mapq=30  \
	--min-flen=50  \
	--max-flen=1000  \
	--keep-chrm=TRUE  \
	--keep-single=FALSE  \
	--keep-secondary=False  \
	--overwrite=True  \
	--max-num=20000  \
	--min-cov=5  \
	--verbose=True

snaptools snap-add-bmat \
	--snap-file=merged.sort.snap \
	--bin-size-list 1000 5000 10000 \
	--verbose=True
