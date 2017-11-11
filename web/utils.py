import csv


def allowed_file(filename):
    ALLOWED_EXTENSIONS = ['csv', 'txt', 'tsv']
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extension(filename):
    return filename.split('.')[-1].lower()


def extract_docs(filename):
    e = extension(filename)
    if e == 'txt':
        with open(filename, 'rt') as f:
            lines = f.readlines()
            docs = list(map(lambda l: l.strip(), lines))
    else:
        with open(filename, 'rt', newline='') as f:
            docs = []
            reader = csv.reader(f)
            for row in reader:
                docs.append(row[-1])

    return docs
