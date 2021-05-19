library(statnet)

samplk1 = matrix(c(
0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,
1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,
1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,
0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,
1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,
1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,
0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,
0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,
1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,
1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,
1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,
1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0), 18,18, byrow=TRUE)

samplk2 = matrix(c(
0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,
1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,
1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,
1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,
0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,
0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,
0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,
1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,
1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,
1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,
0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,
0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0), 18,18, byrow=TRUE)


samplk3 = matrix(c(
0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,
1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,
1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,
0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,
0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,
0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,
0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,
0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,
0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,
0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,
0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,
1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,
1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,
0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,
0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,
0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0), 18, 18, byrow=TRUE)


samplk = list()
samplk[[1]] = as.network(samplk1, directed=TRUE)
samplk[[2]] = as.network(samplk2, directed=TRUE)
samplk[[3]] = as.network(samplk3, directed=TRUE)



stergm.fit.edgeonly.samplk = stergm(samplk, 
    formation = ~edges,
    dissolution = ~edges,
    # times=c(1,2),
    estimate='CMLE')

summary(stergm.fit.edgeonly.samplk)

# #time 2-3
# f: -2.6784
# d: 0.8557
# #time 1-2
# f: -2.3427
# d: 0.5596
# #no time arg
# f: -2.4980
# d: 0.7066