# pcalg
install.packages("BiocManager")
# igraph is required by ggm
install.packages("igraph")
library(igraph)
BiocManager::install(c("graph", "RBGL", "ggm"), ask = FALSE)
install.packages("pcalg")
# test installation
library(pcalg)

install.packages(c("ggplot2", "xtable"))
