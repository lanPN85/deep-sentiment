import os
import re

EXCLUDE = re.compile('(\t)|(<[a-z]{1,2} ?/?>)|(\n)')
files = os.listdir('aclImdb/train/unsup/')
fout = open('imdb.txt', 'wt')
for fl in files:
    f = open('aclImdb/train/unsup/' + fl, 'rt', encoding='utf-8')
    lines = ' '.join(f.readlines())
    lines = re.sub(EXCLUDE, ' ', lines)
    print(lines)
    fout.write(lines + '\n')
    f.close()

fout.close()
