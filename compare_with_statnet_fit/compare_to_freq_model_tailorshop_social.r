library(statnet)

KAPFTS1 = matrix(c(
0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0), 39, 39, byrow=TRUE)

KAPFTS2 = matrix(c(
0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1,
0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0), 39, 39, byrow=TRUE)

net1 = as.network(KAPFTS1, directed=FALSE)
net2 = as.network(KAPFTS2, directed=FALSE)
tailor_social = list()
tailor_social[[1]] = net1
tailor_social[[2]] = net2


stergm.fit1.tailor_social = stergm(tailor_social, 
    formation = ~edges,
    dissolution = ~edges,
    time=c(1,2),
    estimate='CMLE')

summary(stergm.fit1.tailor_social) #-2.5611, 0.8806 // -0.1880,0.5129
fit1.gof <- gof(stergm.fit1.tailor_social)
# fit1.gof <- gof(stergm.fit1.tailor_social, coef=c(10, 0))
#bstergm: formation [-2.7086, 0.9838], dissolution [-2.7814, 1.0281]
# print(fit1.gof)
par(mfrow=c(2,4))
plot(fit1.gof, plotlogodds=FALSE)


net.formation = net1|net2
# ergm.formation.fit = ergm(net.formation~edges, constraints=~atleast(net1)) #-1.3502
ergm.formation.fit = ergm(net.formation~edges+gwesp(0.25, fixed=TRUE), constraints=~atleast(net1))
summary(ergm.formation.fit) #-2.5621, 0.8827
ergm.formation.gof = gof(ergm.formation.fit)
par(mfrow=c(1,4))
plot(ergm.formation.gof, plotlogodds=FALSE)

net.dissolution = net1&net2
# ergm.dissolution.fit = ergm(net.dissolution~edges, constraints=~atmost(net1)) #0.6274
ergm.dissolution.fit = ergm(net.dissolution~edges+gwesp(0.25, fixed=TRUE), constraints=~atmost(net1))
summary(ergm.dissolution.fit) #-0.1878, 0.5118
ergm.dissolution.gof = gof(ergm.dissolution.fit)
par(mfrow=c(1,4))
plot(ergm.dissolution.gof, plotlogodds=FALSE)


par(mfrow=c(4,4))
plot(ergm.formation.gof, plotlogodds=FALSE)
plot(fit1.gof, plotlogodds=FALSE)
plot(ergm.dissolution.gof, plotlogodds=FALSE)


#참고용(bergm 비교용)
ergm.net1.fit = ergm(net1~edges+gwesp(0.25, fixed=TRUE))
summary(ergm.net1.fit)

ergm.net2.fit = ergm(net2~edges+gwesp(0.25, fixed=TRUE))
summary(ergm.net2.fit)
mcmc.diagnostics(ergm.net2.fit)

ergm.netf.fit = ergm(net.formation~edges+gwesp(0.25, fixed=TRUE))
summary(ergm.netf.fit)


ergm.netd.fit = ergm(net.dissolution~edges+gwesp(0.25, fixed=TRUE))
summary(ergm.netd.fit)

#with time c(1,2) argument
# ============================== 
# Summary of formation model fit
# ==============================
# Formula:   ~edges + gwdegree(0.25, fixed = TRUE) + gwesp(0.25, fixed = TRUE)

# Call:
# ergm(formula = formation, constraints = constraints.form, offset.coef = offset.coef.form,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.form,
#     verbose = verbose)

# Iterations:  2 out of 20 

# Monte Carlo MLE Results:
#                  Estimate Std. Error MCMC % z value Pr(>|z|)
# edges             -2.5761     1.2362      0  -2.084   0.0372 *
# gwdeg.fixed.0.25 -18.7338     7.6373      0  -2.453   0.0142 *
# gwesp.fixed.0.25   0.8973     0.9109      0   0.985   0.3246
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 808.2  on 583  degrees of freedom
#  Residual Deviance: 591.1  on 580  degrees of freedom

# AIC: 597.1    BIC: 610.2    (Smaller is better.)

# ================================
# Summary of dissolution model fit
# ================================

# Formula:   ~edges + gwdegree(0.25, fixed = TRUE) + gwesp(0.25, fixed = TRUE)

# Call:
# ergm(formula = dissolution, constraints = constraints.diss, offset.coef = offset.coef.diss,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.diss, 
#     verbose = verbose)

# Iterations:  2 out of 20

# Monte Carlo MLE Results:
#                   Estimate Std. Error MCMC % z value Pr(>|z|)
# edges            -0.190885   0.428352      0  -0.446   0.6559
# gwdeg.fixed.0.25  0.000973   0.897453      0   0.001   0.9991
# gwesp.fixed.0.25  0.513791   0.239462      0   2.146   0.0319 *
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 219.0  on 158  degrees of freedom
#  Residual Deviance: 197.6  on 155  degrees of freedom

# AIC: 203.6    BIC: 212.8    (Smaller is better.)


#without time arg
# ==============================
# Summary of formation model fit 
# ==============================

# Formula:   ~edges + gwdegree(0.25, fixed = TRUE) + gwesp(0.25, fixed = TRUE)

# Call:
# ergm(formula = formation, constraints = constraints.form, offset.coef = offset.coef.form,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.form,
#     verbose = verbose)

# Iterations:  4 out of 20

# Monte Carlo MLE Results:
#                  Estimate Std. Error MCMC % z value Pr(>|z|)  
# edges             -2.6026     1.1610      0  -2.242   0.0250 *
# gwdeg.fixed.0.25 -17.0754    10.3180      0  -1.655   0.0979 .
# gwesp.fixed.0.25   0.9165     0.8520      0   1.076   0.2820
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 808.2  on 583  degrees of freedom
#  Residual Deviance: 591.1  on 580  degrees of freedom

# AIC: 597.1    BIC: 610.2    (Smaller is better.)

# ================================
# Summary of dissolution model fit
# ================================

# Formula:   ~edges + gwdegree(0.25, fixed = TRUE) + gwesp(0.25, fixed = TRUE)

# Call:
# ergm(formula = dissolution, constraints = constraints.diss, offset.coef = offset.coef.diss,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.diss,
#     verbose = verbose)

# Iterations:  2 out of 20

# Monte Carlo MLE Results:
#                   Estimate Std. Error MCMC % z value Pr(>|z|)
# edges            -0.183027   0.426393      0  -0.429   0.6677
# gwdeg.fixed.0.25 -0.003001   0.908293      0  -0.003   0.9974
# gwesp.fixed.0.25  0.511887   0.239557      0   2.137   0.0326 *
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 219.0  on 158  degrees of freedom
#  Residual Deviance: 197.7  on 155  degrees of freedom

# AIC: 203.7    BIC: 212.9    (Smaller is better.)


stergm.fit2.tailor_social = stergm(tailor_social, 
    formation = ~edges+gwdegree(0.25, fixed=TRUE)+gwesp(0.25, fixed=TRUE)+gwdsp(0.25, fixed=TRUE),
    dissolution = ~edges+gwdegree(0.25, fixed=TRUE)+gwesp(0.25, fixed=TRUE)+gwdsp(0.25, fixed=TRUE),
    estimate='CMLE', verbose=TRUE) #CMLE verbose 놓고 돌려보자

summary(stergm.fit2.tailor_social)
# ==============================
# Summary of formation model fit 
# ==============================

# Formula:   ~edges + gwdegree(0.25, fixed = TRUE) + gwesp(0.25, fixed = TRUE) +
#     gwdsp(0.25, fixed = TRUE)

# Call:
# ergm(formula = formation, constraints = constraints.form, offset.coef = offset.coef.form,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.form,
#     verbose = verbose)

# Iterations:  20 out of 20

# Monte Carlo MLE Results:
#                    Estimate Std. Error MCMC %  z value Pr(>|z|)    
# edges            -2.350e+00  1.238e+00      0   -1.898   0.0577 .
# gwdeg.fixed.0.25 -1.264e+04  6.780e+01    100 -186.384   <1e-04 ***
# gwesp.fixed.0.25  8.210e-01  9.292e-01      0    0.884   0.3769
# gwdsp.fixed.0.25  7.990e-02  1.537e-01      0    0.520   0.6032
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance:   808.2  on 583  degrees of freedom
#  Residual Deviance: 25402.2  on 579  degrees of freedom

# AIC: 25410    BIC: 25428    (Smaller is better.)

# ================================
# Summary of dissolution model fit
# ================================

# Formula:   ~edges + gwdegree(0.25, fixed = TRUE) + gwesp(0.25, fixed = TRUE) +
#     gwdsp(0.25, fixed = TRUE)

# Call:
# ergm(formula = dissolution, constraints = constraints.diss, offset.coef = offset.coef.diss,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.diss, 
#     verbose = verbose)

# Iterations:  2 out of 20

# Monte Carlo MLE Results:
#                  Estimate Std. Error MCMC % z value Pr(>|z|)
# edges             0.17852    0.54428      0   0.328   0.7429
# gwdeg.fixed.0.25  0.26155    0.86851      0   0.301   0.7633
# gwesp.fixed.0.25  0.53004    0.23883      0   2.219   0.0265 *
# gwdsp.fixed.0.25 -0.07981    0.06678      0  -1.195   0.2321  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 219.0  on 158  degrees of freedom
#  Residual Deviance: 196.2  on 154  degrees of freedom

# AIC: 204.2    BIC: 216.5    (Smaller is better.)



stergm.fit3.tailor_social = stergm(tailor_social, 
    formation = ~edges+gwesp(0.25, fixed=TRUE),
    dissolution = ~edges+gwesp(0.25, fixed=TRUE),
    estimate='CMLE') #CMLE

summary(stergm.fit3.tailor_social)
# ==============================
# Summary of formation model fit
# ==============================

# Formula:   ~edges + gwesp(0.25, fixed = TRUE)

# Call:
# ergm(formula = formation, constraints = constraints.form, offset.coef = offset.coef.form,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.form,
#     verbose = verbose)

# Iterations:  2 out of 20

# Monte Carlo MLE Results:
#                  Estimate Std. Error MCMC % z value Pr(>|z|)  
# edges             -2.6004     1.2241      0  -2.124   0.0336 *
# gwesp.fixed.0.25   0.9076     0.9003      0   1.008   0.3134
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 808.2  on 583  degrees of freedom
#  Residual Deviance: 591.3  on 581  degrees of freedom

# AIC: 595.3    BIC: 604.1    (Smaller is better.)

# ================================
# Summary of dissolution model fit
# ================================

# Formula:   ~edges + gwesp(0.25, fixed = TRUE)

# Call:
# ergm(formula = dissolution, constraints = constraints.diss, offset.coef = offset.coef.diss,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.diss,
#     verbose = verbose)

# Iterations:  2 out of 20

# Monte Carlo MLE Results:
#                  Estimate Std. Error MCMC % z value Pr(>|z|)  
# edges             -0.1943     0.3680      0  -0.528   0.5975
# gwesp.fixed.0.25   0.5179     0.2223      0   2.329   0.0198 *
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 219.0  on 158  degrees of freedom
#  Residual Deviance: 197.8  on 156  degrees of freedom

# AIC: 201.8    BIC: 207.9    (Smaller is better.)

stergm.fit4.tailor_social = stergm(tailor_social, 
    formation = ~edges+gwesp(0.25, fixed=TRUE)+gwdsp(0.25, fixed=TRUE),
    dissolution = ~edges+gwesp(0.25, fixed=TRUE)+gwdsp(0.25, fixed=TRUE),
    estimate='CMLE') #CMLE

summary(stergm.fit4.tailor_social)
# ==============================
# Summary of formation model fit
# ==============================

# Formula:   ~edges + gwesp(0.25, fixed = TRUE) + gwdsp(0.25, fixed = TRUE)

# Call:
# ergm(formula = formation, constraints = constraints.form, offset.coef = offset.coef.form,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.form,
#     verbose = verbose)

# Iterations:  7 out of 20

# Monte Carlo MLE Results:
#                  Estimate Std. Error MCMC % z value Pr(>|z|)    
# edges            -2.99616    1.25699      0  -2.384 0.017144 *
# gwesp.fixed.0.25  1.34537    0.94452      0   1.424 0.154334
# gwdsp.fixed.0.25 -0.23871    0.06558      0  -3.640 0.000273 ***
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 808.2  on 583  degrees of freedom
#  Residual Deviance: 585.1  on 580  degrees of freedom

# AIC: 591.1    BIC: 604.2    (Smaller is better.) 

# ================================
# Summary of dissolution model fit
# ================================

# Formula:   ~edges + gwesp(0.25, fixed = TRUE) + gwdsp(0.25, fixed = TRUE)

# Call:
# ergm(formula = dissolution, constraints = constraints.diss, offset.coef = offset.coef.diss,
#     eval.loglik = eval.loglik, estimate = ergm.estimate, control = control$CMLE.control.diss,
#     verbose = verbose)

# Iterations:  2 out of 20

# Monte Carlo MLE Results:
#                  Estimate Std. Error MCMC % z value Pr(>|z|)
# edges             0.22094    0.52774      0   0.419   0.6755
# gwesp.fixed.0.25  0.49755    0.21766      0   2.286   0.0223 *
# gwdsp.fixed.0.25 -0.07524    0.06601      0  -1.140   0.2543
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#      Null Deviance: 219.0  on 158  degrees of freedom
#  Residual Deviance: 196.1  on 155  degrees of freedom

# AIC: 202.1    BIC: 211.3    (Smaller is better.)


stergm.fit5.tailor_social = stergm(tailor_social, 
    formation = ~edges + gwdegree(0.25, fixed=TRUE), # gwdsp(0.25, fixed=TRUE), #+gwdsp(0.25, fixed=TRUE),
    dissolution = ~edges + gwdegree(0.25, fixed=TRUE), # gwdsp(0.25, fixed=TRUE), #+gwdsp(0.25, fixed=TRUE),
    estimate='CMLE') #CMLE

summary(stergm.fit5.tailor_social)