library(statnet)


name_vec = c("m3.19", "w3.19", "f3.19")

sequence = list()
for(i in 1:length(name_vec)){
    name = name_vec[i]
    net.structure = read.table(paste("BayesianSTERGM/data/",name,".network.txt",sep=''), sep=",")
    sequence[[i]] = as.network(as.matrix(net.structure), directed=FALSE)    
}


stergm.fit.seq = stergm(sequence, 
    formation= ~edges+gwesp(0.5,fixed=TRUE),
    dissolution= ~edges+gwesp(0.5,fixed=TRUE),
    estimate="CMLE"
    )
summary(stergm.fit.seq)
