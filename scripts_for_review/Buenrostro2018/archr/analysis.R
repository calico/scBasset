library(ArchR)
library(gtools)

input_files <- "/home/yuanh/sc_basset/datasets/Buenrostro_2018/raw/merged.sort.bam"
addArchRGenome("hg19")
addArchRThreads(threads = 8) 

# create arrow files
ArrowFiles <- createArrowFiles(
  inputFiles = input_files,
  sampleNames = "Buenrostro2018",
  minTSS = 0, #Dont set this too high because you can always increase later
  minFrags = 0, 
  maxFrags = 1e+20,
  bcTag= "RG",
  addTileMat = TRUE,
  addGeneScoreMat = FALSE,
)

# create project
proj <- ArchRProject(
  ArrowFiles = ArrowFiles, 
  outputDirectory = "Buenrostro2018",
  copyArrows = TRUE
)

head(proj@cellColData)

# filter cells
ad <- readRDS("../cistopic/ad_atac.rds")
proj <- subsetCells(proj, cellNames = paste0("Buenrostro2018#", ad$cell$cell))

# save and load the project
saveArchRProject(proj, overwrite=T, load=T)

############
# QC plots #
############

# TSS enrichment boxplot
p1 <- plotGroups(
  ArchRProj = proj, 
  groupBy = "Sample", 
  colorBy = "cellColData", 
  name = "TSSEnrichment",
  plotAs = "violin",
  alpha = 0.4,
  addBoxPlot = TRUE
)
dir.create("Buenrostro2018/Plots")
pdf("Buenrostro2018/Plots/QC_TSSenrich.pdf", 8, 6)
print(p1)
dev.off()

# nFrag boxplot
p2 <- plotGroups(
  ArchRProj = proj, 
  groupBy = "Sample", 
  colorBy = "cellColData", 
  name = "log10(nFrags)",
  plotAs = "violin",
  alpha = 0.4,
  addBoxPlot = TRUE
)
pdf("Buenrostro2018/Plots/QC_log10_nFrags.pdf", 8, 6)
print(p2)
dev.off()

# save and load the project
saveArchRProject(proj, overwrite=T, load=T)

#######
# LSI #
#######
proj <- loadArchRProject("Buenrostro2018/")

# LSI
proj <- addIterativeLSI(
    ArchRProj = proj,
    useMatrix = "TileMatrix", 
    name = "IterativeLSI", 
    iterations = 2, 
    clusterParams = list( #See Seurat::FindClusters
        resolution = c(0.2), 
        sampleCells = 10000, 
        n.start = 10
    ))

m <- getReducedDims(proj, reducedDims = "IterativeLSI")
write.csv(m, file="embed.csv")