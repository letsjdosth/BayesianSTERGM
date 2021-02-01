library(tergm)

data(samplk)

# Fit a transition from Time 1 to Time 2
samplk12 <- stergm(list(samplk1, samplk2, samplk3), #<-여기를 바꿔가며 돌릴 것
                    formation=~edges+mutual+transitiveties+cyclicalties,
                    dissolution=~edges+mutual+transitiveties+cyclicalties,
                    estimate="CMLE")

samplk12.gof <- gof(samplk12)

samplk12.gof

summary(samplk12.gof)
par(mfrow=c(2,5))
plot(samplk12.gof)

plot(samplk12.gof, plotlogodds=TRUE)
