library(statnet)

test_structure1 = matrix(c(
    0,1,1,0,0,0,0,0,0,0,
    1,0,1,1,0,0,0,0,0,0,
    1,1,0,1,0,0,0,0,0,0,
    0,1,1,0,1,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0), 10, 10, byrow=TRUE)

test_structure2 = matrix(c(
    0,1,1,0,0,0,0,0,0,0,
    1,0,1,1,0,0,0,0,0,0,
    1,1,0,1,0,0,0,0,0,0,
    0,1,1,0,1,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,
    0,0,0,0,0,0,1,0,0,1,
    0,0,0,0,0,0,1,0,0,1,
    0,0,0,0,0,0,0,1,1,0), 10, 10, byrow=TRUE)

test_structure3 =  matrix(c(
    0,0,1,0,0,0,0,0,1,1,
    0,0,1,1,0,0,0,0,1,0,
    1,1,0,0,0,0,0,1,0,0,
    0,1,0,0,1,1,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,
    0,0,1,0,0,0,1,0,0,1,
    1,1,0,0,0,0,1,0,0,1,
    1,0,0,0,0,0,0,1,1,0), 10, 10, byrow=TRUE)


net1 = as.network(test_structure1, directed=FALSE)
net2 = as.network(test_structure2, directed=FALSE)
net3 = as.network(test_structure3, directed=FALSE)

test_nets = list()
test_nets[[1]] = net1
test_nets[[2]] = net2
test_nets[[3]] = net3


summary(net1~edges+gwesp(0.25, fixed=TRUE))
summary(net2~edges+gwesp(0.25, fixed=TRUE))
summary(net3~edges+gwesp(0.25, fixed=TRUE))

fit1 = stergm(test_nets, 
    formation = ~edges+gwesp(0.25, fixed=TRUE),
    dissolution = ~edges+gwesp(0.25, fixed=TRUE),
    estimate='CMLE')

summary(fit1)
#edgeonly -1.962, 2.0149
#edge+gwesp(0.25) -1.6782 -0.2836 // 2.7845, -0.4699