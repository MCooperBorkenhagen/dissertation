

packages = read.csv('requirements.txt', stringsAsFactors = F)[[1]]
lapply(packages, require, ch = T)