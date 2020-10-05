#traceplot
posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BERGMrun/2_bergmPosteriorSample.csv", header=FALSE)
mean(posterior_sample$V1)
mean(posterior_sample$V2)
par(mfrow=c(1,2))
plot(1:nrow(posterior_sample), posterior_sample$V1, type="l")
plot(1:nrow(posterior_sample), posterior_sample$V2, type="l")
# acf(posterior_sample$V1)
# acf(posterior_sample$V2)


network_diag_stats = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BERGMrun/2_lastExNetSamplerNetworkStats.csv", header=TRUE)

# for(i in 1:length(network_diag_stats)){
length(network_diag_stats) #34
# names(network_diag_stats)

par(mfrow=c(2,6))
for(i in 1:12){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}

par(mfrow=c(2,6))
for(i in 13:24){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}

par(mfrow=c(2,6))
for(i in 25:34){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}