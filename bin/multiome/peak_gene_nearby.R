#!/usr/bin/Rscript

library(rtracklayer)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)
gene_name <- as.character(args[1])

# additional params
peak_search_window <- 400000 # peaks within +/- 200kb of TSS
gene_search_window <- 800000  # genes within +/- 400kb of TSS
out_dir <- "peaks"
gtf <- import("/home/yuanh/programs/genomes/mm10/mm10.refGene.gtf", format="gtf")
bed <- import.bed("/home/yuanh/analysis/kidney_kenwhite/old_analysis/CKD_aug_10/scbasset_atac/processed/peaks.bed")

print(sprintf('find peaks within 200kb from TSS of %s', gene_name))
dir.create(out_dir)

# GR for all genes
annot <- data.table(data.frame(mcols(gtf)))
annot$seqname <- as.character(seqnames(gtf))
annot$start <- start(gtf)
annot$end <- end(gtf)
annot$strand <- as.character(strand(gtf))
genes <- unique(annot[, list(start = min(start), end = max(end), symbol = gene_id), by = list(gene_id, seqname, strand)])
genesGR <- GRanges(seqnames = genes$seqname,
                   ranges = IRanges(start = genes$start,end = genes$end),
                   strand = genes$strand,
                   symbol = genes$symbol
                  )
g_gr <- genesGR[genesGR$symbol==gene_name]
end(g_gr) <- start(g_gr)

tss_400k <- resize(g_gr, width=peak_search_window, fix="center")
tss_800k <- resize(g_gr, width=gene_search_window, fix="center")

##########################
# peaks within +/- 200kb #
##########################
names(bed) <- paste0("peak_", 1:length(bed))
ovlp <- findOverlaps(bed, tss_400k, type='any')
peaks <- bed[queryHits(ovlp)]

##########################
# genes within +/- 400kb #
##########################
gene_bed <- sort(genesGR[subjectHits(findOverlaps(tss_800k, genesGR, ignore.strand=T))], ignore.strand=T)
out <- data.frame(chr=seqnames(gene_bed), 
                  start=start(gene_bed), 
                  end=end(gene_bed), 
                  gene=mcols(gene_bed))
write.csv(out, file=sprintf("%s/genes_nearby.csv", out_dir))

#############
# distances #
#############
a <- gene_bed[strand(gene_bed)=='+']
end(a) <- start(a)
b <- gene_bed[strand(gene_bed)=='-']
start(b) <- end(b)
gene_bed_tss <- sort(c(a, b), ignore.strand=T)
names(gene_bed_tss) <- gene_bed_tss$symbol

dist_m <- matrix(0, nrow=length(peaks), ncol=length(gene_bed_tss), 
                 dimnames=list(names(peaks), names(gene_bed_tss)))

# distance to Kl TSS
# negative: peak is upstream of gene TSS
# positive: peak is downstream of gene TSS
for (i in names(peaks)) {
    for (j in names(gene_bed_tss)) {
        p <- peaks[i]
        g <- gene_bed_tss[j]
        
        peak_gene_dist <- distance(p, g, ignore.strand=T)
        if (end(p) < start(g)) {
            peak_gene_dist <- (-peak_gene_dist)
        }
        dist_m[i, j] <- peak_gene_dist
    }
}
export.bed(peaks, sprintf("%s/peaks.bed", out_dir))
write.csv(dist_m, sprintf("%s/peaks_gene_dist.csv", out_dir))

