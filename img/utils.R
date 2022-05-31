library(ggplot2)
library(xtable)

WIDTH <- 5
HEIGHT <- 3

BASELINE <- c(
  "NSigma",
  "SPOT",
  "DFS",
  "DFS-MS",
  "DFS-MH",
  "RW-Par",
  "RW-2",
  "ENMF",
  "CRD"
)

dumpPlot <- function(filename, g, width = WIDTH, height = HEIGHT, ...) {
  ggsave(filename, g, width = width, height = height, ...)
}

addStyle <- function(g) {
  g + theme(
    panel.background = element_blank(),
    panel.grid.major = element_line(colour = "grey"),
    axis.line = element_line(colour = "black")
  )
}

drawBlank <- function(d, xlab, ylab, logx = FALSE, logy = FALSE) {
  g <- ggplot(d, aes(x = x, y = y)) +
    labs(x = xlab, y = ylab)
  if (logx) {
    g <- g + scale_x_log10()
  }
  if (logy) {
    g <- g + scale_y_log10()
  }
  addStyle(g)
}

formatFloat <- function(d, digits = 3) {
  formatC(d, digits = digits, format = "f")
}

findGraphName <- function(graphMethod) {
  if (startsWith(graphMethod, "PC_gauss")) {
    "PC-guass"
  } else if (startsWith(graphMethod, "PC_gsq")) {
    "PC-gsq"
  } else if (startsWith(graphMethod, "PCTS")) {
    "PCTS"
  } else if (!grepl("_", graphMethod, fixed = TRUE)) {
    graphMethod
  } else {
    print(sprintf("Unknown graph: %s", graphMethod))
    graphMethod
  }
}

findRankName <- function(items) {
  size <- length(items)
  if (startsWith(items[1], "RHT")) {
    if (size == 1) {
      "RHT"
    } else if (size == 2 && items[2] == "PG") {
      "RHT-PG"
    }  else if (size == 2 && items[2] == "DA") {
      "CIRCA"
    } else {
      print(sprintf("Unknown method: %s", paste(items, collapse = "-")))
      "Unknown"
    }
  } else if (size == 1) {
    if (startsWith(items[1], "SPOT")) {
      "SPOT"
    } else if (startsWith(items[1], "ENMF")) {
      "ENMF"
    } else if (startsWith(items[1], "CRD")) {
      "CRD"
    } else {
      items[1]
    }
  } else if (startsWith(items[2], "DFS")) {
    if (size == 2) {
      "DFS"
    } else if (items[3] == "Pearson") {
      "DFS-MS"
    } else {
      print(sprintf("Unknown method: %s", paste(items, collapse = "-")))
      "Unknown"
    }
  } else if (startsWith(items[2], "MicroHECL")) {
    "DFS-MH"
  } else if (startsWith(items[2], "RW_2_")) {
    "RW-2"
  } else if (startsWith(items[2], "RW_")) {
    "RW-Par"
  } else {
    rankName <- paste(items, collapse = "-")
    rankName
  }
}

extendMethod <- function(d) {
  items <- strsplit(d$method, "-")
  d$graphMethod <- sapply(items, function (item) {
    item[1]
  })
  d$graphName <- sapply(d$graphMethod, findGraphName)

  items <- lapply(items, function (item) {
    item[2:length(item)]
  })
  d$rankMethod <- sapply(items, function (item) {
    paste(item, collapse = "-")
  })
  d$rankName <- sapply(items, findRankName)
  d
}

chooseBest <- function(filename, n = 2) {
  RANK_NAMES <- c(
    BASELINE,
    "RHT",
    "CIRCA"
  )
  d <- read.csv(filename)
  d <- extendMethod(d)
  d$rankName <- as.factor(d$rankName)
  report <- data.frame()
  for (rankName in RANK_NAMES) {
    methodData <- d[d$rankName == rankName, ]
    methodData <- methodData[order(methodData$AC.5, methodData$Avg.5, decreasing = TRUE),]
    report <- rbind(report, head(methodData, n = n))
  }
  for (rankName in levels(d$rankName)) {
    if (!(rankName %in% RANK_NAMES)) {
      methodData <- d[d$rankName == rankName, ]
      methodData <- methodData[order(methodData$AC.5, methodData$Avg.5, decreasing = TRUE),]
      report <- rbind(report, head(methodData, n = n))
    }
  }
  report
}

formatChooseBest <- function(filename, outputFilename, ...) {
  report <- chooseBest(filename, ...)
  report <- data.frame(
    Method = report$rankName,
    "AC@1" = formatFloat(report$AC.1),
    "AC@3" = formatFloat(report$AC.3),
    "AC@5" = formatFloat(report$AC.5),
    "Avg@5" = formatFloat(report$Avg.5),
    "duration" = formatFloat(report$avg.duration),
    name = report$method
  )
  print(xtable(report), file = outputFilename, include.rownames = FALSE)
}
