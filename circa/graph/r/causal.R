library(pcalg)

runPC <- function(d, CItest = "gauss", alpha = 0.05, m.max = Inf, numCores = 1) {
  if (CItest == "gsq") {
    suffStat <- list(dm = d, adaptDF = TRUE)
    indepTest <- disCItest
  } else {
    # gauss
    stopifnot(CItest == "gauss")
    suffStat <- list(C = cor(d), n = nrow(d))
    indepTest <- gaussCItest
  }
  g <- pc(suffStat = suffStat, indepTest = indepTest, alpha = alpha,
          p = ncol(d), m.max = m.max, numCores = numCores,
          skel.method = "stable.fast", maj.rule = TRUE, solve.confl = TRUE)
  as(g@graph, "matrix")
}
