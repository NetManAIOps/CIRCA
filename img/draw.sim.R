source("img/utils.R")

simSummary <- function(dirName, outputFilename) {
  columns <- c("AC.1", "AC.3", "AC.5", "avg.duration")
  numGraph <- 10
  reportList <- lapply(c(50, 100, 500), function(size) {
    dfList <- lapply(0:(numGraph - 1), function(index) {
      read.csv(sprintf("%s/report-%d-%d.csv", dirName, size, index))
    })
    avgReport(dfList, columns=columns)
  })

  methods <- reportList[[1]]$method
  reportList <- lapply(reportList, function(d) {
    index <- sapply(d$method, function(k) {which(k == methods)})
    d[index, ]
  })
  report <- extendMethod(data.frame(method = methods))
  report <- data.frame(method = report$rankName)
  for (r in reportList) {
     report <- cbind(report, r[, columns])
  }
  print(xtable(report), file = outputFilename, include.rownames = FALSE)
}

simSignificance <- function(dirName) {
  numGraph <- 10
  target <- "Sim-SRCA"
  baselines <- c(
    "NSigma",
    "SPOT",
    "DFS",
    "DFS-MS",
    "DFS-MH",
    "ENMF",
    "CRD",
    "RW-2",
    "RW-Par",
    "SRCA"
  )
  columns <- c("AC.1", "AC.3", "AC.5")
  for (size in c(50, 100, 500)) {
    report <- data.frame()
    for (index in 0:(numGraph - 1)) {
      report <- rbind(report, read.csv(sprintf("%s/report-%d-%d.csv", dirName, size, index)))
    }
    report <- extendMethod(report)
    report$rankName <- as.factor(report$rankName)
    significance <- data.frame(method = baselines)
    for (column in columns) {
      targetValues <- report[report$rankName == target, column]
      significance[, column] <- sapply(baselines, function(method) {
        t.test(targetValues, report[report$rankName == method, column], alternative = "greater")$p.value
      })
    }
    print(sprintf("p-value of t-test for %s over baselines with size %d", target, size))
    print(significance)
    targetValues <- report[report$rankName == target, "AC.1"]
    print(t.test(targetValues, report[report$rankName == "DFS", "AC.1"], alternative = "greater"))
  }
}

compareFaults <- function(dirName, outputFilename) {
  intensities <- c("weak", "mixed", "strong")
  dfList <- lapply(intensities, function(intensity) {
    chooseBest(sprintf("%s/%s.csv", dirName, intensity), n = 1)
  })
  methods <- dfList[[1]]$method
  report <- extendMethod(data.frame(method = methods))
  report <- data.frame(method = report$rankName)
  for (d in dfList) {
    index <- sapply(d$method, function(k) {which(k == methods)})
    report <- cbind(report, data.frame(
      "AC@1" = formatFloat(d$AC.1),
      "AC@5" = formatFloat(d$AC.5)
    ))
  }
  print(xtable(report), file = outputFilename, include.rownames = FALSE)
}

formatChooseBest("report/sim-tuning/report.csv", "img/output/best-sim-tuning.tex", n = 3)
simSummary("report/sim", "img/output/summary-sim.tex")
simSignificance("report/sim")
compareFaults("report/sim-robustness/50", "img/output/report-sim-robustness-50.tex")
