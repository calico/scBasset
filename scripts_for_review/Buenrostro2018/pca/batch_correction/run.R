library(kBET)
library(lisi)
library(harmony)

ad <- readRDS("../../cistopic/ad.rds")
embed <- read.csv("../projection.csv", header=T, row.names=1)

embed_bc <- HarmonyMatrix(embed, ad$cells, "batch", do_pca=FALSE)
write.csv(embed_bc, file="projection_bc.csv")
