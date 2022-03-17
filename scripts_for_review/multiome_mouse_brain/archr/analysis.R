library(ArchR)
library(gtools)
library(gridExtra)

input_files <- "/home/yuanh/sc_basset/datasets/10x_ARC_mouse_brain/raw/e18_mouse_brain_fresh_5k_atac_fragments.tsv.gz"
addArchRGenome("mm10")
addArchRThreads(threads = 8) 

# create arrow files
ArrowFiles <- createArrowFiles(
  inputFiles = input_files,
  sampleNames = "ATAC",
  minTSS = 0, #Dont set this too high because you can always increase later
  minFrags = 0, 
  maxFrags = 1e+20,
  addTileMat = TRUE,
  addGeneScoreMat = TRUE
)

# create project
proj <- ArchRProject(
  ArrowFiles = ArrowFiles, 
  outputDirectory = "mouse_proj",
  copyArrows = TRUE
)

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
dir.create("mouse_proj/Plots")
pdf("mouse_proj/Plots/QC_TSSenrich.pdf", 8, 6)
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
pdf("mouse_proj/Plots/QC_log10_nFrags.pdf", 8, 6)
print(p2)
dev.off()

# filter cells
ad <- readRDS("../cistopic/ad_atac.rds")
proj <- subsetCells(proj, cellNames = paste0("ATAC#", ad$cells$cell_id))

# save and load the project
saveArchRProject(proj, overwrite=T, load=T)

#######
# LSI #
#######
proj <- loadArchRProject("mouse_proj/")

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
write.csv(m, file="projection.csv")