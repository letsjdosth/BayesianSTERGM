namevec = c("net1seq_chain1", "net2seq_chain1")
name = namevec[2]

# ==================================================================================

formation_posterior_sample = read.csv(paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_formation.csv",sep=''), header=FALSE)
formation_posterior_sample=formation_posterior_sample[10000:dim(formation_posterior_sample)[1],1]
formation_posterior_sample=formation_posterior_sample[seq(1, length(formation_posterior_sample), by = 10)]
write.table(formation_posterior_sample, paste("BayesianSTERGM/candidate_run/", name,"_aftercut_BSTERGM_formation.csv", sep=""), sep=",", row.names=FALSE, col.names=FALSE)

traceplot_filename = paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_formation_traceplot.png",sep="")
hist_filename = paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_formation_histogram.png",sep="")
acf_filename = paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_formation_acf.png",sep="")
plot(1:length(formation_posterior_sample), formation_posterior_sample, type="l", xlab="iteration", ylab="theta_formation")
dev.copy(png,filename=traceplot_filename)
dev.off()

hist(formation_posterior_sample, xlab="theta_formation", main="",prob=TRUE, breaks=40)
dev.copy(png,filename=hist_filename)
dev.off()

acf(formation_posterior_sample, main="")
dev.copy(png,filename=acf_filename)
dev.off()



dissolution_posterior_sample = read.csv(paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_dissolution.csv",sep=''), header=FALSE)
dissolution_posterior_sample=dissolution_posterior_sample[10000:dim(dissolution_posterior_sample)[1],1]
dissolution_posterior_sample=dissolution_posterior_sample[seq(1, length(dissolution_posterior_sample), by = 10)]
write.table(dissolution_posterior_sample, paste("BayesianSTERGM/candidate_run/", name,"_aftercut_BSTERGM_dissolution.csv", sep=""), sep=",", row.names=FALSE, col.names=FALSE)

traceplot_filename = paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_dissolution_traceplot.png",sep="")
hist_filename = paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_dissolution_histogram.png",sep="")
acf_filename = paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_dissolution_acf.png",sep="")
plot(1:length(dissolution_posterior_sample), dissolution_posterior_sample, type="l", xlab="iteration", ylab="theta_dissolution")
dev.copy(png,filename=traceplot_filename)
dev.off()

hist(dissolution_posterior_sample, xlab="theta_dissolution", main="",prob=TRUE, breaks=40)
dev.copy(png,filename=hist_filename)
dev.off()

acf(dissolution_posterior_sample, main="")
dev.copy(png,filename=acf_filename)
dev.off()



network_diag_stats = read.csv(paste("BayesianSTERGM/candidate_run/",name,"_BSTERGM_lastExNetSamplerNetworkStats.csv",sep=''), header=FALSE)

# for(i in 1:length(network_diag_stats)){
length(network_diag_stats) #11
# names(network_diag_stats)
# mean(network_diag_stats$userSpecific0)
# mean(network_diag_stats$userSpecific1)
# par(mfrow=c(1,2))
ylabname= c("num of edges", "GWESP(0.5)")
for(i in 1:1){
    diagfilename = paste("BayesianSTERGM/candidate_run/",name,"_lastsampler_",ylabname[i],".png",sep="")
    plot(1:nrow(network_diag_stats), network_diag_stats[,i], type="l", xlab="iter", ylab=ylabname[i])
    dev.copy(png,filename=diagfilename)
    dev.off()
}




