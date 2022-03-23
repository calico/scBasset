library(universalmotif)
library(Biostrings)

motifs <- read_meme("/home/yuanh/programs/source/motif_databases.12.18/CIS-BP/Homo_sapiens.meme")
new_names <-  sapply(motifs, function(x) {x@altname})
names(motifs) <- gsub("\\(|\\).*", "", new_names)

for (i in 1:length(motifs)) {
    write.csv(motifs[[i]]@motif, paste0("motif_pwms/", names(motifs)[i], ".csv"))
}

