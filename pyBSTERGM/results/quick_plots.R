posterior_chain0 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_0chain.csv", header=FALSE)
posterior_chain1 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_1chain.csv", header=FALSE)
posterior_chain2 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_2chain.csv", header=FALSE)
posterior_chain3 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_3chain.csv", header=FALSE)
posterior_chain4 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_4chain.csv", header=FALSE)
posterior_chain5 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_5chain.csv", header=FALSE)
posterior_chain6 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_6chain.csv", header=FALSE)
posterior_chain7 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_7chain.csv", header=FALSE)


#for each chain
posterior_sample = posterior_chain0 #set chain
mean(posterior_sample$V1)
mean(posterior_sample$V2)
mean(posterior_sample$V3)
mean(posterior_sample$V4)
par(mfrow=c(4,1))
plot(1:nrow(posterior_sample), posterior_sample$V1, type="l")
plot(1:nrow(posterior_sample), posterior_sample$V2, type="l")
plot(1:nrow(posterior_sample), posterior_sample$V3, type="l")
plot(1:nrow(posterior_sample), posterior_sample$V4, type="l")
par(mfrow=c(4,2))
hist(posterior_sample$V1)
acf(posterior_sample$V1)
hist(posterior_sample$V2)
acf(posterior_sample$V2)
hist(posterior_sample$V3)
acf(posterior_sample$V3)
hist(posterior_sample$V4)
acf(posterior_sample$V4)

#for each parameter
par(mfrow=c(4,2))
plot(1:nrow(posterior_chain0), posterior_chain0$V1, type="l") #V1/V2/V3/V4 선택
plot(1:nrow(posterior_chain1), posterior_chain1$V1, type="l")
plot(1:nrow(posterior_chain2), posterior_chain2$V1, type="l")
plot(1:nrow(posterior_chain3), posterior_chain3$V1, type="l")
plot(1:nrow(posterior_chain4), posterior_chain4$V1, type="l")
plot(1:nrow(posterior_chain5), posterior_chain5$V1, type="l")
plot(1:nrow(posterior_chain6), posterior_chain6$V1, type="l")
plot(1:nrow(posterior_chain7), posterior_chain7$V1, type="l")
par(mfrow=c(4,2))
hist(posterior_chain0$V1)
hist(posterior_chain1$V1)
hist(posterior_chain2$V1)
hist(posterior_chain3$V1)
hist(posterior_chain4$V1)
hist(posterior_chain5$V1)
hist(posterior_chain6$V1)
hist(posterior_chain7$V1)


#netstat
networkStat_chain0 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_0chain_statNet.csv", header=FALSE)
networkStat_chain1 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_1chain_statNet.csv", header=FALSE)
networkStat_chain2 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_2chain_statNet.csv", header=FALSE)
networkStat_chain3 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_3chain_statNet.csv", header=FALSE)
networkStat_chain4 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_4chain_statNet.csv", header=FALSE)
networkStat_chain5 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_5chain_statNet.csv", header=FALSE)
networkStat_chain6 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_6chain_statNet.csv", header=FALSE)
networkStat_chain7 = read.csv("C:/gitProject/BayesianSTERGM/pyBSTERGM/results/seq1_7chain_statNet.csv", header=FALSE)

netStat_sample = networkStat_chain1 #set chain
mean(netStat_sample$V1) #edge
mean(netStat_sample$V2) #GWESP(0.5)
par(mfrow=c(2,1))
plot(1:nrow(netStat_sample), netStat_sample$V1, type="l")
plot(1:nrow(netStat_sample), netStat_sample$V2, type="l")


