import csv


result=[]
with open('net3seq_chain1_aftercut_BSTERGM_formation.csv', newline='') as csvfile:
# with open('net3seq_chain1_aftercut_BSTERGM_dissolution.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csv_reader:
        result.append(float(row[0]))

print(result)
print(len(result))
