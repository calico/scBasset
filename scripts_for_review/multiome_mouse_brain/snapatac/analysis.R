library(SnapATAC)
library(GenomicRanges)

# filter barcodes
x.sp = createSnap(
    file="e18_mouse_brain_fresh_5k_atac_fragments.snap",
    sample="mouse_brain_multiome",
    num.cores=8
  )

ad <- readRDS("../cistopic/ad_atac.rds")
barcodes <- ad$cells$cell_id
x.sp = x.sp[which(x.sp@barcode %in% barcodes),]

showBinSizes("e18_mouse_brain_fresh_5k_atac_fragments.snap")
x.sp = addBmatToSnap(x.sp, bin.size=5000, num.cores=8)
x.sp = makeBinary(x.sp, mat="bmat") #binarize (following tutorial)

# remove blacklist
black_list = read.table("/home/yuanh/genomes/mm10/mm10.blacklist.bed.gz");
black_list.gr = GRanges(
    black_list[,1], 
    IRanges(black_list[,2], black_list[,3])
)
idy = queryHits(findOverlaps(x.sp@feature, black_list.gr))
if(length(idy) > 0){x.sp = x.sp[,-idy, mat="bmat"]}
x.sp

# remove chrM
chr.exclude = seqlevels(x.sp@feature)[grep("random|chrM", seqlevels(x.sp@feature))]
idy = grep(paste(chr.exclude, collapse="|"), x.sp@feature)
if(length(idy) > 0){x.sp = x.sp[,-idy, mat="bmat"]}
x.sp

# remove 5%
bin.cov = log10(Matrix::colSums(x.sp@bmat)+1)
bin.cutoff = quantile(bin.cov[bin.cov > 0], 0.95)
idy = which(bin.cov <= bin.cutoff & bin.cov > 0)
x.sp = x.sp[, idy, mat="bmat"]
x.sp

# diffusion map for dimension reduction
x.sp = runDiffusionMaps(
    obj=x.sp,
    input.mat="bmat", 
    num.eigs=50
)
embed <- x.sp@smat@dmat

# reorder rows
rownames(embed) <- x.sp@barcode
embed <- embed[ad$cells$cell_id, ]
write.csv(embed, file="projection.csv")
