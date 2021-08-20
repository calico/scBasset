library(Biostrings)
library(rtracklayer)
library(BSgenome.Hsapiens.UCSC.hg38)
source("get_seqs.R")
dir.create("Homo_sapiens_motif_fasta")

#######################################
# generate dinucleotide shuffled seqs #
#######################################
# we used command line script fasta_ushuffle to generate shuffled sequences.
bed <- import.bed("/home/yuanh/analysis/sc_basset/datasets/10x_ARC_PBMC/peaks.bed")
bed <- resize(bed, width=1344, fix="center")
set.seed(10)
examples <- sample(bed, 1000)
seqs <- get.seqs(BSgenome.Hsapiens.UCSC.hg38, examples, 1)
writeXStringSet(seqs, "Homo_sapiens_motif_fasta/example_peaks.fasta", format="fasta", width=1344) # fasta_ushuffle requires sequence in 1 line.
cmd <- "fasta_ushuffle -k 2 < Homo_sapiens_motif_fasta/example_peaks.fasta > Homo_sapiens_motif_fasta/shuffled_peaks.fasta"
system(cmd)
    
###############
# read motifs #
###############
library(universalmotif)
library(Biostrings)
motifs <- read_meme("/home/yuanh/programs/source/motif_databases.12.18/CIS-BP/Homo_sapiens.meme")
shuffled_pks <- readDNAStringSet("Homo_sapiens_motif_fasta/shuffled_peaks.fasta", format="fasta")

#######################
# write fasta + motif #
#######################
dir.create("Homo_sapiens_motif_fasta/shuffled_peaks_motifs/")
for (i in 1:length(motifs)) {
  tf <- gsub("\\(|\\)","", strsplit(motifs[[i]]@altname, "_")[[1]][1])
  pwm <- motifs[[i]]@motif
  set.seed(10)
  out <- apply(pwm, 2, function(x) {
    return(sample(rownames(pwm), 1000, replace=T, prob=x))
  })
  motif_seqs <- apply(out, 1, function(x) paste(x, collapse=""))
  
  # insert motif seqs to the center of shuffled peaks
  left_coord <- width(shuffled_pks)[1]/2 - floor(ncol(pwm)/2)
  left <- as.character(subseq(shuffled_pks, start=1, end=left_coord))
  right <- as.character(subseq(shuffled_pks, start=left_coord+ncol(pwm)+1, end=width(shuffled_pks)[1]))
  shuffled_pks_motifs <- DNAStringSet(paste0(left, motif_seqs, right))
  writeXStringSet(shuffled_pks_motifs, paste0("Homo_sapiens_motif_fasta/shuffled_peaks_motifs/", tf, ".fasta"), format="fasta")
  print(i)
}
