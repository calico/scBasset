library(SnapATAC)
library(GenomicRanges)

# filter barcodes
x.sp = createSnap(
    file="merged.sort.snap",
    sample="buenrostro",
    num.cores=8
  )

ad <- readRDS("../cistopic/ad_atac.rds")
barcodes <- toupper(ad$cells$cell)
x.sp = x.sp[which(x.sp@barcode %in% barcodes),]

showBinSizes("merged.sort.snap")
x.sp = addBmatToSnap(x.sp, bin.size=5000, num.cores=8)
x.sp = makeBinary(x.sp, mat="bmat") #binarize (following tutorial)
 
# remove blacklist
black_list = read.table("/home/yuanh/genomes/hg19/hg19-blacklist.bed.gz");
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


#hist(
#    bin.cov[bin.cov > 0], 
#    xlab="log10(bin cov)", 
#    main="log10(Bin Cov)", 
#    col="lightblue", 
#    xlim=c(0, 5)
#  )

# diffusion map for dimension reduction
x.sp = runDiffusionMaps(
    obj=x.sp,
    input.mat="bmat", 
    num.eigs=50
)

pdf("choose_eigen.pdf", 8, 8)
plotDimReductPW(
    obj=x.sp, 
    eigs.dims=1:50,
    point.size=0.3,
    point.color="grey",
    point.shape=19,
    point.alpha=0.6,
    down.sample=5000,
    pdf.file.name=NULL, 
    pdf.height=7, 
    pdf.width=7
  )
dev.off()

embed <- x.sp@smat@dmat

# reorder rows
rownames(embed) <- x.sp@barcode
embed <- embed[barcodes, ]
write.csv(embed, file="embed.csv")
