library(cisTopic)

#############
# run model #
#############
ad <- readRDS("ad.rds")
m <- ad$matrix
m@x <- rep(1, length(m@i)) # binarize the data
colnames(m) <- ad$cells$cell_id
rownames(m) <- paste0(ad$genes$chr,":",ad$genes$start,"-",ad$genes$end)
cisTopicObject <- createcisTopicObject(m)
cisTopicObject <- runCGSModels(cisTopicObject, topic=c(2, 5, 10, 20, 30, 40, 50, 60, 80, 100), seed=987, nCores=10, burnin = 120, iterations = 200, addModels=FALSE)
saveRDS(cisTopicObject, file="cistopicmodel.rds")

###############
# selet model #
###############
library(cisTopic)
n_topics <- c(2, 5, 10, 20, 30, 40, 50, 60, 80, 100)
cisTopicObject <- readRDS("cistopicmodel.rds")

# some evaluations
pdf("select_models.pdf", 10, 5)
par(mfrow=c(1,2))
cisTopicObject <- selectModel(cisTopicObject)
dev.off()

pdf("convergence.pdf", 5, 5)
logLikelihoodByIter(cisTopicObject, select=n_topics)
dev.off()

# save all models
dir.create("models")
for (i in 1:length(n_topics)) {
  a <- selectModel(cisTopicObject, select=n_topics[i])
  cellassign <- modelMatSelection(a, 'cell', 'Probability')
  write.csv(t(cellassign), file=sprintf("models/topics_%d.csv", n_topics[i]))
  print(i)
}

