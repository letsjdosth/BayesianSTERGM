#traceplot

formation_posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BSTERGM_emp_data_run_edgeonly/BSTERGM_formation.csv", header=FALSE)
mean(formation_posterior_sample$V1)
mean(formation_posterior_sample$V2)
par(mfrow=c(1,2))
plot(1:nrow(formation_posterior_sample), formation_posterior_sample$V1, type="l")
plot(1:nrow(formation_posterior_sample), formation_posterior_sample$V2, type="l")
par(mfrow=c(1,2))
acf(formation_posterior_sample$V1)
acf(formation_posterior_sample$V2)




dissolution_posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BSTERGM_emp_data_run_edgeonly/BSTERGM_dissolution.csv", header=FALSE)
mean(dissolution_posterior_sample$V1)
mean(dissolution_posterior_sample$V2)
par(mfrow=c(1,2))
plot(1:nrow(dissolution_posterior_sample), dissolution_posterior_sample$V1, type="l")
plot(1:nrow(dissolution_posterior_sample), dissolution_posterior_sample$V2, type="l")
par(mfrow=c(1,2))
acf(dissolution_posterior_sample$V1)
acf(dissolution_posterior_sample$V2)



network_diag_stats = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BSTERGM_emp_data_run_edgeonly/BSTERGM_lastExNetSamplerNetworkStats.csv", header=TRUE)

# for(i in 1:length(network_diag_stats)){
length(network_diag_stats) #11
# names(network_diag_stats)
mean(network_diag_stats$userSpecific0)
mean(network_diag_stats$userSpecific1)
par(mfrow=c(1,2))
for(i in 1:2){
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=names(network_diag_stats)[i])
}
