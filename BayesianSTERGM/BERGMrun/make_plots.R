#traceplot
posterior_sample = read.csv("C:/gitProject/BayesianSTERGM/BayesianSTERGM/BERGMrun/0_bergmPosteriorSample.csv", header=FALSE)
mean(posterior_sample$V1)
mean(posterior_sample$V2)
par(mfrow=c(1,2))
plot(1:nrow(posterior_sample), posterior_sample$V1, type="l")
plot(1:nrow(posterior_sample), posterior_sample$V2, type="l")
# acf(posterior_sample$V1)
# acf(posterior_sample$V2)

