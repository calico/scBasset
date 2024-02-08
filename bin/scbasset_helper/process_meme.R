library(optparse)

#############
# arguments #
#############
option_list = list(
    make_option(c("-s", "--seed"), type="integer", default=10, 
                help="random seed"),
    
    make_option(c("-n", "--num"), type="integer", default=1000, 
              help="number of peaks to sample"),

    make_option(c("-o", "--outdir"), type="character", default="motif_seqs", 
              help="output sequence folder"),
    
    make_option(c("-g", "--genome"), type="character", default=NULL, 
                help="must provide genome build, supports: mm9, mm10, hg19, hg38."),
    
    make_option(c("-b", "--bed"), type="character", default=NULL, 
                help="must provide bed file from scbasset preprocess."),
    
    make_option(c("-m", "--meme"), type="character", default=NULL, 
                help="must provide meme file.")
)

# pass arguments
opt <- parse_args(OptionParser(option_list=option_list))
seed <- opt$seed
outdir <- opt$outdir
n_peaks <- opt$num
genome <- opt$genome
bed_file <- opt$bed
meme_file <- opt$meme

# check valid
if (is.null(opt$bed)){
    stop("must provide bed file.", call.=FALSE)
}
if (is.null(opt$meme)){
    stop("must provide meme file.", call.=FALSE)
}
if (!genome %in% c("mm9", "mm10", "hg19", "hg38")){
    stop("only support mm9, mm10, hg19, hg38.", call.=FALSE)
}

dir.create(outdir)
set.seed(seed)

library(Biostrings)
library(rtracklayer)
library(parallel)
library(universalmotif)
source("~/programs/myscripts/NGS_scripts/get_seqs.R")

#######################################
# generate dinucleotide shuffled seqs #
#######################################
bed <- import.bed(bed_file)
bed <- resize(bed, width=1344, fix="center")
examples <- sample(bed, n_peaks)

# load corresponding genome
if (genome=="mm9") {
    library(BSgenome.Mmusculus.UCSC.mm9)
    seqs <- get.seqs(BSgenome.Mmusculus.UCSC.mm9, examples, 1)
}
if (genome=="mm10") {
    library(BSgenome.Mmusculus.UCSC.mm10)
    seqs <- get.seqs(BSgenome.Mmusculus.UCSC.mm10, examples, 1)
}
if (genome=="hg19") {
    library(BSgenome.Hsapiens.UCSC.hg19)
    seqs <- get.seqs(BSgenome.Hsapiens.UCSC.hg19, examples, 1)
}
if (genome=="hg38") {
    library(BSgenome.Hsapiens.UCSC.hg38)
    seqs <- get.seqs(BSgenome.Hsapiens.UCSC.hg38, examples, 1)
}

writeXStringSet(seqs, sprintf("%s/example_peaks.fasta", outdir), format="fasta", width=1344) # fasta_ushuffle requires sequence in 1 line.
cmd <- sprintf("fasta_ushuffle -k 2 < %s/example_peaks.fasta > %s/shuffled_peaks.fasta", outdir, outdir)
system(cmd)

###############
# read motifs #
###############
motifs <- read_meme(meme_file)
shuffled_pks <- readDNAStringSet(sprintf("%s/shuffled_peaks.fasta",outdir), format="fasta")

###############
# plot motifs #
###############
dir.create(sprintf("%s/motif_logos/", outdir))
dir.create(sprintf("%s/motif_pwms/", outdir))
for (i in 1:length(motifs)) {
    
    motif_name <- motifs[[i]]@name
    
    # save PWM
    write.csv(motifs[[i]]@motif, sprintf('%s/motif_pwms/%s.csv',outdir, motif_name))

    # plot seqLogo
    png(sprintf('%s/motif_logos/%s.png', outdir, motif_name), 900, 200)
    print(view_motifs(motifs[[i]]))
    dev.off()
    
    if (i%%100==1){print(i)}
}

#######################
# write fasta + motif #
#######################
dir.create(sprintf("%s/shuffled_peaks_motifs/", outdir))
for (i in 1:length(motifs)) {
    
    motif_name <- motifs[[i]]@name
    
    pwm <- motifs[[i]]@motif
    out <- apply(pwm, 2, function(x) {
        return(sample(rownames(pwm), 1000, replace=T, prob=x))
    })
    
    motif_seqs <- apply(out, 1, function(x) {paste(x, collapse="")})
  
    # insert motif seqs to the center of shuffled peaks
    left_coord <- width(shuffled_pks)[1]/2 - floor(ncol(pwm)/2)
    left <- as.character(subseq(shuffled_pks, start=1, end=left_coord))
    right <- as.character(subseq(shuffled_pks, start=left_coord+ncol(pwm)+1, end=width(shuffled_pks)[1]))
    shuffled_pks_motifs <- DNAStringSet(paste0(left, motif_seqs, right))
    
    writeXStringSet(shuffled_pks_motifs, sprintf('%s/shuffled_peaks_motifs/%s.fasta', outdir, motif_name), format="fasta")
    if (i%%100==1){print(i)}
}
