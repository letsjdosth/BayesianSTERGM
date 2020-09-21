#traceplot
par(mfrow=c(1,2))
formation_posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/bergmPosteriorSample.csv", header=FALSE)
mean(formation_posterior_sample$V1)
plot(1:nrow(formation_posterior_sample), formation_posterior_sample$V1, type="l")
plot(1:nrow(formation_posterior_sample), formation_posterior_sample$V2, type="l")
acf(formation_posterior_sample$V1)
acf(formation_posterior_sample$V2)

par(mfrow=c(1,2))
dissolution_posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/dissolution.csv", header=FALSE)
plot(1:nrow(dissolution_posterior_sample), dissolution_posterior_sample$V1, type="l")
plot(1:nrow(dissolution_posterior_sample), dissolution_posterior_sample$V2, type="l")


