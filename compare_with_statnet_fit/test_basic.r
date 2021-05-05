library(statnet)

test_structure = matrix(c(0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,1,0), 5,5)
test_network = as.network(test_structure, directed=FALSE)
plot(test_network)

summary(test_network~edges+dsp(0)+dsp(1)+dsp(2)+dsp(3)+dsp(4))
summary(test_network~edges+gwdsp(0.3, fixed=TRUE))
summary(test_network~edges+gwdegree(0.3, fixed=TRUE))


# test_structure = matrix(c(
#     0,1,1,0,0,
#     1,0,0,0,1,
#     1,1,0,1,0,
#     0,0,0,0,1,
#     1,0,0,1,0),5,5,byrow=T)
# test_network = as.network(test_structure)
# plot(test_network)

# summary(test_network~edges+odegree(0)+odegree(1)+odegree(2)+odegree(3)+odegree(4)) #5.393469
# summary(test_network~edges+mutual+transitiveties++cyclicalties)
# summary(test_network~edges+ctriple+ttriple)
# summary(test_network~edges+istar(2)+ostar(2))


fit = ergm(test_network~gwesp)
?gwesp
?gwdsp




# elem1 = matrix(c(
#         0,1,1,0,0,0,0,0,0,0,
#         1,0,1,1,0,0,0,0,0,0,
#         1,1,0,1,0,0,0,0,0,0,
#         0,1,1,0,1,0,0,0,0,0,
#         0,0,0,1,0,0,0,0,0,0,
#         0,0,0,0,0,0,0,0,0,0,
#         0,0,0,0,0,0,0,0,0,0,
#         0,0,0,0,0,0,0,0,0,0,
#         0,0,0,0,0,0,0,0,0,0,
#         0,0,0,0,0,0,0,0,0,0),10,10)
# net1 = as.network(elem1, directed=FALSE)
# fit1 = ergm(net1 ~ edges + gwesp(0.5, fixed=TRUE))
# mcmc.diagnostics(fit1)


# elem2 = matrix(c(
#         0,1,1,0,0,0,0,0,0,0,
#         1,0,1,1,0,0,0,0,0,0,
#         1,1,0,1,0,0,0,0,0,0,
#         0,1,1,0,1,1,0,0,0,0,
#         0,0,0,1,0,0,0,0,0,0,
#         0,0,0,1,0,0,0,0,0,0,
#         0,0,0,0,0,0,0,1,1,0,
#         0,0,0,0,0,0,1,0,0,1,
#         0,0,0,0,0,0,1,0,0,1,
#         0,0,0,0,0,0,0,1,1,0),10,10)
# net2 = as.network(elem2, directed=FALSE)
# fit2 = ergm(net2 ~ edges + gwesp(0.5, fixed=TRUE))
# mcmc.diagnostics(fit2)

# elem3 = matrix(c(
#         0,0,1,0,0,0,0,0,1,1,
#         0,0,1,1,0,0,0,0,1,0,
#         1,1,0,0,0,0,0,1,0,0,
#         0,1,0,0,1,1,0,0,0,0,
#         0,0,0,1,0,0,0,0,0,0,
#         0,0,0,1,0,0,0,0,0,0,
#         0,0,0,0,0,0,0,1,1,0,
#         0,0,1,0,0,0,1,0,0,1,
#         1,1,0,0,0,0,1,0,0,1,
#         1,0,0,0,0,0,0,1,1,0),10,10)
# net3 = as.network(elem3, directed=FALSE)
# fit3 = ergm(net3 ~ edges + gwesp(0.5, fixed=TRUE))
# mcmc.diagnostics(fit3)
