# for ArchR: merge and sort by position
samtools merge merged.bam sc-bams_nodup/*.bam
samtools sort -o merged.sort.bam merged.bam

