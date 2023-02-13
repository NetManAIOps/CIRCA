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

install_packages <- function(pkgs, ...) {
  if (os == "osx") {
      install.packages(pkgs, type = "mac.binary", ...)
  } else if (os == "windows") {
      install.packages(pkgs, type = "win.binary", ...)
  } else {
      install.packages(pkgs, ...)
  }
}

# pcalg
install_packages("BiocManager")
# igraph is required by ggm
install_packages("igraph")
library(igraph)
BiocManager::install(c("graph", "RBGL", "ggm"), ask = FALSE)
install_packages("pcalg")
# test installation
library(pcalg)

install_packages(c("ggplot2", "xtable"))
