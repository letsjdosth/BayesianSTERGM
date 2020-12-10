

network_diag_stats = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BSTERGM_lastExNetSamplerNetworkStats.csv", header=TRUE)

# for(i in 1:length(network_diag_stats)){
length(network_diag_stats) #11
# names(network_diag_stats)
mean(network_diag_stats$userSpecific0)
mean(network_diag_stats$userSpecific1)
par(mfrow=c(1,2))
for(i in 1:2){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}






formation_posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BSTERGM_formation.csv", header=FALSE)
par(mfrow=c(1,2))
plot(1:nrow(formation_posterior_sample), formation_posterior_sample$V1, type="l")
plot(1:nrow(formation_posterior_sample), formation_posterior_sample$V2, type="l")



dissolution_posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BSTERGM_dissolution.csv", header=FALSE)
par(mfrow=c(1,2))
plot(1:nrow(dissolution_posterior_sample), dissolution_posterior_sample$V1, type="l")
plot(1:nrow(dissolution_posterior_sample), dissolution_posterior_sample$V2, type="l")


