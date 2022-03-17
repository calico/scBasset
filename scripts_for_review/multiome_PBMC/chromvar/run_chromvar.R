# chromvar tutorial
library(chromVAR)
library(motifmatchr)
library(BSgenome.Hsapiens.UCSC.hg38)
library(SummarizedExperiment)
library(BiocParallel)
register(MulticoreParam(16))

ad <- readRDS("../cistopic/ad_atac.rds")
m <- ad$matrix  
m@x <- rep(1, length(m@i))
gr <- GRanges(seqnames=ad$genes$chr, 
              ranges=IRanges(ad$genes$start, ad$genes$end), 
              strand="*")
se <- SummarizedExperiment(assays=list(counts=m),
                           rowRanges=gr, 
                           colData=ad$cells)
# compute gc bias
se <- addGCBias(se, genome = BSgenome.Hsapiens.UCSC.hg38)

# find motif hits
motifs <- getJasparMotifs(species="Homo sapiens")
motif_ix <- matchMotifs(motifs, se, genome = BSgenome.Hsapiens.UCSC.hg38)
kmer_ix <- matchKmers(6, se,  genome = BSgenome.Hsapiens.UCSC.hg38)

# compute z-scores
dev_motif <- computeDeviations(object = se, annotations = motif_ix)
saveRDS(dev_motif, "dev_motif.rds")
write.csv(t(deviationScores(dev_motif)), "z_motif.csv")
dev_kmer <- computeDeviations(object = se, annotations = kmer_ix)
saveRDS(dev_kmer, "dev_kmer.rds")
write.csv(t(deviationScores(dev_kmer)), "z_kmer.csv")

