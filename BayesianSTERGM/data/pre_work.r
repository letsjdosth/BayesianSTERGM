#f1.19
f1.19.ss = read.table("BayesianSTERGM/data/rawdata/w3.19.ss.txt")
dim(f1.19.ss) #29,29
f1.19.sf = read.table("BayesianSTERGM/data/rawdata/w3.19.sf.txt")
dim(f1.19.sf) #29,23

# 29+23=52
f1.19.mat = matrix(0,52,52)
f1.19.mat[1:29, 1:29] = as.matrix(f1.19.ss)
f1.19.mat[1:29, 30:52] = as.matrix(f1.19.sf)
f1.19.mat[30:52, 1:29] = t(as.matrix(f1.19.sf))
# f1.19.mat

# library(statnet)
f1.19.net = as.network.matrix(f1.19.mat, directed=FALSE)
plot(f1.19.net)
f1.19.netMat = as.matrix.network(f1.19.net)
# print(f1.19.netMat)
write.table(f1.19.netMat, "BayesianSTERGM/data/w3.19.network.txt")