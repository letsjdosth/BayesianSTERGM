#traceplot
posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BERGMrun/3_bergmPosteriorSample.csv", header=FALSE)
mean(posterior_sample$V1)
mean(posterior_sample$V2)
par(mfrow=c(1,2))
plot(1:nrow(posterior_sample), posterior_sample$V1, type="l")
plot(1:nrow(posterior_sample), posterior_sample$V2, type="l")

acf(posterior_sample$V1)
acf(posterior_sample$V2)


network_diag_stats = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BERGMrun/3_lastExNetSamplerNetworkStats.csv", header=TRUE)

# for(i in 1:length(network_diag_stats)){
length(network_diag_stats) #33
# names(network_diag_stats)
par(mfrow=c(1,2))
for(i in 32:33){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}

par(mfrow=c(2,4))
for(i in 1:8){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}

par(mfrow=c(2,4))
for(i in 9:16){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}

par(mfrow=c(2,4))
for(i in 17:24){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}

par(mfrow=c(2,4))
for(i in 25:32){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}