library(cicero)
library(monocle3)

ad <- readRDS("../cistopic/ad.rds")
m <- ad$matrix
m@x <- rep(1, length(m@i))

set.seed(10)
input_cds <-  suppressWarnings(new_cell_data_set(m,
                                                 cell_metadata = ad$cells,
                                                 gene_metadata = ad$genes))
input_cds <- monocle3::detect_genes(input_cds)
input_cds <- detect_genes(input_cds)
input_cds <- estimate_size_factors(input_cds)
input_cds <- preprocess_cds(input_cds, method = "LSI")
write.csv(reducedDims(input_cds)[["LSI"]], file="LSI.csv")
