get_os <- function() {
  os <- .Platform$OS.type
  if (os == "unix") {
    sysinf <- Sys.info()
    if (!is.null(sysinf)) {
      if (sysinf["sysname"] == "Darwin") {
        os <- "osx"
      }
      if (startsWith(sysinf["machine"], "arm")) {
        os <- sprintf("%s-%s", os, sysinf["machine"])
      }
    } else if (grepl("^darwin", R.version$os)) {
      os <- "osx"
    }
  } # else "windows"
  os
}
os <- get_os()

# pcalg
install.packages("BiocManager")
# igraph is required by ggm
if (os == "osx") {
    install.packages("igraph", type = "mac.binary")
} else if (os == "windows") {
    install.packages("igraph", type = "win.binary")
} else {
    install.packages("igraph")
}
library(igraph)
BiocManager::install(c("graph", "RBGL", "ggm"), ask = FALSE)
install.packages("pcalg")
# test installation
library(pcalg)

install.packages(c("ggplot2", "xtable"))
