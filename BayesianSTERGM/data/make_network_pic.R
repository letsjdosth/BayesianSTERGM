library(statnet)

name_vec = c("f1.19", "f2.19", "f3.19", "m1.19", "m2.19", "m3.19", "w1.19", "w2.19", "w3.19")


for(i in 1:length(name_vec)){
    name = name_vec[i]
    net.structure = read.table(paste("BayesianSTERGM/data/",name,".network.txt",sep=''), sep=",")
    net = as.network(as.matrix(net.structure), directed=FALSE)
    plot(net)
    filename = paste("BayesianSTERGM/data/",name,"_nework.png",sep="")
    dev.copy(png,filename=filename)
    dev.off()
}
