library(statnet)
name_vec = c("f1.19", "f2.19", "f3.19", "m1.19", "m2.19", "m3.19", "w1.19", "w2.19", "w3.19")

for(i in 1:9){
    name = name_vec[i]
    data_ss = read.table(paste("BayesianSTERGM/data/rawdata/",name,".ss.txt", sep=""))
    # dim(data_ss) #29,29
    data_sf = read.table(paste("BayesianSTERGM/data/rawdata/",name,".sf.txt", sep=""))
    # dim(data_sf) #29,23

    # 29+23=52
    data_mat = matrix(0,52,52)
    data_mat[1:29, 1:29] = as.matrix(data_ss)
    data_mat[1:29, 30:52] = as.matrix(data_sf)
    data_mat[30:52, 1:29] = t(as.matrix(data_sf))
    # data_mat

    
    net = as.network.matrix(data_mat, directed=FALSE)
    netMat = as.matrix.network(net)
    write.table(netMat, paste("BayesianSTERGM/data/", name,".network.txt", sep=""), sep=",", row.names=FALSE, col.names=FALSE)
}

