import os
import random
import time
import re

EXCLUDE = re.compile('(\t)|(<[a-z]{1,2} ?/?>)|(\n)')


def doc_txt(file):
    f = open(file, "r", encoding="utf8")
    x = f.read()
    f.close()
    return re.sub(EXCLUDE, ' ', str(x))


path_train_neg = "./aclImdb/train/neg/"
path_train_pos = "./aclImdb/train/pos/"
path_test_neg = "./aclImdb/test/neg/"
path_test_pos = "./aclImdb/test/pos/"


#khoi dau
list_train_pos = os.listdir(path_train_pos)
list_train_neg = os.listdir(path_train_neg)
list_test_pos = os.listdir(path_test_pos)
list_test_neg = os.listdir(path_test_neg)

#tiep tuc
m = len(list_train_pos)

current = time.time()
for i in range(m):
    
    list_train_pos[i] = "Positive\t" + doc_txt(path_train_pos + list_train_pos[i])
    list_train_neg[i] = "Negative\t"+ doc_txt(path_train_neg + list_train_neg[i])
    list_test_pos[i] = "Positive\t" + doc_txt(path_test_pos + list_test_pos[i])
    list_test_neg[i] = "Negative\t" + doc_txt(path_test_neg + list_test_neg[i])

current = time.time() - current
print(current)
print("dang them du lieu")
#them du lieu cho file train
a = []
k = m*7//10

a.extend(list_train_pos[:k])
a.extend(list_train_neg[:k])
a.extend(list_test_pos[:k])
a.extend(list_test_neg[:k])

random.shuffle(a)
lena = len(a)
print(lena)
f = open("train.tsv","w", encoding="utf8")
for i in range(lena):
    f.write(a[i]+"\n")

f.close()


###them du lieu cho file test
b = []
x = k + m*15//100
b.extend(list_train_pos[k:x])
b.extend(list_train_neg[k:x])
b.extend(list_test_pos[k:x])
b.extend(list_test_neg[k:x])
##
random.shuffle(b)
lenb = len(b)
print(lenb)
f = open("test.tsv","w", encoding = "utf8")
for i in range(lenb):
    f.write(b[i]+"\n")

f.close()
#
#
###them du lieu cho file val
##
c = []
c.extend(list_train_pos[x:])
c.extend(list_train_neg[x:])
c.extend(list_test_pos[x:])
c.extend(list_test_neg[x:])
#
##
random.shuffle(c)
lenc = len(c)
print(lenc)
f = open("val.tsv","w", encoding="utf8")
for i in range(lenc):
    f.write(c[i]+"\n")

f.close()








