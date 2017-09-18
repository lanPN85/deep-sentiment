import os
train1 = open("train.ft.txt", "r", encoding='utf-8')
test1 = open("test.ft.txt", "r", encoding='utf-8')
total1 = open("total.txt", "w+", encoding='utf-8')
lines1 = train1.readlines()
lines2 = test1.readlines()
for line in lines1:
    total1.write(line)
for line in lines2:
    total1.write(line)
train1.close()
test1.close()
total1.close()


total = open("total.txt", "r", encoding='utf-8')
train = open("train.tsv", "w+", encoding='utf-8')
test = open("test.tsv", "w+", encoding='utf-8')
val = open("val.tsv", "w+", encoding='utf-8')
lines = total.readlines()

pos_count = 0
neg_count = 0

for line in lines:
    if line[:10] == "__label__2":
        strr = "Positive\t" + line[11:]
        if pos_count < 1400000:
            train.write(strr)
        elif pos_count < 1700000:
            test.write(strr)
        else:
            val.write(strr)
        pos_count += 1
    else:
        strr = "Negative\t" + line[11:]
        if neg_count < 1400000:
            train.write(strr)
        elif neg_count < 1700000:
            test.write(strr)
        else:
            val.write(strr)
        neg_count += 1
total.close()
train.close()
test.close()
val.close()

os.remove("total.txt")
