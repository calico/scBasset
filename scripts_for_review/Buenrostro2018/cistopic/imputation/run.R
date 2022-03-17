library(cisTopic)

cisTopicObject <- readRDS("../cistopicmodel.rds")

# use 40 topics
a <- selectModel(cisTopicObject, select=40)
pred.matrix <- predictiveDistribution(a)

write.csv(pred.matrix, file="imputed.csv")