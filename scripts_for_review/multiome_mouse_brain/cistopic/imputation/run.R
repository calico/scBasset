library(cisTopic)

cisTopicObject <- readRDS("../cistopicmodel.rds")

# use 60 topics
a <- selectModel(cisTopicObject, select=60)
pred.matrix <- predictiveDistribution(a)

write.csv(pred.matrix, file="imputed.csv")