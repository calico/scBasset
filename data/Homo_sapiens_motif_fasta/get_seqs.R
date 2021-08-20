#' Get sequences for genomic reanges
#'
#' @param org BSGenome object of the organism
#' @param regions GenomicRanges object of the co ordinates
#' @param no.cores Number of threads for parallel processing. Default 3
#' @param metadat.field Metadata field name for naming sequences
#' @param fasta.file Name of the fasta.file in which sequences need to written
#' @return DNAStringSet object of sequences

get.seqs <- function (org, regions, no.cores=1) {

  ## Function for chromosome
  get.seq.chr <- function (chr) {
    seq <- org[[chr]]
    if (class (seq) == 'MaskedDNAString')
      seq <- unmasked (seq)

    ## Get all sequences
    chr.regions <- regions[which (as.character (seqnames (regions)) == chr)]
    region.seq <- DNAStringSet (Views (seq, start=start (chr.regions), end=end (chr.regions)))
    ## Reverse complement negative strands
    rev <- as.logical (strand (chr.regions) == '-')
    region.seq[rev] <- reverseComplement (region.seq[rev])
    names (region.seq) <- values (chr.regions)$.ZZ
    gc ()
    return (region.seq)
  }

  ## Split by chromosomes
  values (regions)$.ZZ <- sprintf ("Unit%d", 1:length (regions))
  seqs <- unique (as.character (seqnames (regions)))

  ## Run in parallel
  all.seqs <- mclapply (seqs, get.seq.chr, mc.cores=no.cores)
  all.seqs <- do.call (c, all.seqs)

  ## Sort to original order
  inds <- sort (as.numeric (gsub ('Unit', '', names (all.seqs))), index.return=TRUE)$ix
  all.seqs <- all.seqs[inds]

  # Clean up
  all.seqs <- as.character (all.seqs)
  gc ()
  all.seqs <- DNAStringSet (all.seqs)
  gc ()
  
  return (all.seqs)
}
  

load.bsgenome <- function (genome) {

  if (!genome %in% c('hg19', 'hg18', 'mm9', 'mm10', 'pFal')) {
    stop ('The specified organism is not supported. Please contact manu@cbio.mskcc.org for adding support to the organism')
  }

  if (genome == 'hg19') {
    library (BSgenome.Hsapiens.UCSC.hg19)
    org <- Hsapiens
  }

  if (genome == 'hg18') {
    library (BSgenome.Hsapiens.UCSC.hg18)
    org <- Hsapiens
  }

  if (genome == 'mm9') {
    library (BSgenome.Mmusculus.UCSC.mm9)
    org <- Mmusculus
  }

  if (genome == 'mm10') {
    library (BSgenome.Mmusculus.UCSC.mm10)
    org <- Mmusculus
  }

  if (genome == 'pFal'){
    library (BSgenome.Pfalciparum.PlasmoDB.plasFalc3D76)
    org <- Pfalciparum
  }

  return (org)
}
