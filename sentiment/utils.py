import csv


def store_dict_csv(path, **kwargs):
    f = open(path, 'w', encoding='utf-8')
    writer = csv.writer(f)
    for key, val in kwargs.items():
        writer.writerow([key, val])
    f.close()
