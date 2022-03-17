library(cisTopic)

cisTopicObject <- readRDS("../cistopicmodel.rds")

# use 80 topics
a <- selectModel(cisTopicObject, select=80)
pred.matrix <- predictiveDistribution(a)

write.csv(pred.matrix, file="imputed.csv")