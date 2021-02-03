library(statnet)
data(samplk)
?samplk
as.matrix(samplk1)
as.matrix(samplk2)
as.matrix(samplk3)


write.table(as.matrix(samplk1), row.names=FALSE, col.names=FALSE, file="pyBSTERGM/samplk1.txt", sep=",")
write.table(as.matrix(samplk2), row.names=FALSE, col.names=FALSE, file="pyBSTERGM/samplk2.txt", sep=",")
write.table(as.matrix(samplk3), row.names=FALSE, col.names=FALSE, file="pyBSTERGM/samplk3.txt", sep=",")

?write.table

# ==========================================

library(xergm.common)
data(knecht)
?knecht
# step 1: make sure the network matrices have node labels
for (i in 1:length(friendship)) {
  rownames(friendship[[i]]) <- 1:nrow(friendship[[i]])
  colnames(friendship[[i]]) <- 1:ncol(friendship[[i]])
}
rownames(primary) <- rownames(friendship[[1]])
colnames(primary) <- colnames(friendship[[1]])
sex <- demographics$sex
names(sex) <- 1:length(sex)

# step 2: imputation of NAs and removal of absent nodes:
friendship <- handleMissings(friendship, na = 10, method = "remove")
friendship <- handleMissings(friendship, na = NA, method = "fillmode")

length(friendship$t1)
length(friendship$t2)
length(friendship$t3)
length(friendship$t4)

write.table(friendship$t1, row.names=FALSE, col.names=FALSE, file="pyBSTERGM/knecht_friendship_t1.txt", sep=",")
write.table(friendship$t2, row.names=FALSE, col.names=FALSE, file="pyBSTERGM/knecht_friendship_t2.txt", sep=",")
write.table(friendship$t3, row.names=FALSE, col.names=FALSE, file="pyBSTERGM/knecht_friendship_t3.txt", sep=",")
write.table(friendship$t4, row.names=FALSE, col.names=FALSE, file="pyBSTERGM/knecht_friendship_t4.txt", sep=",")


# ==========================================
library(RSiena)
