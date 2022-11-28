import csv

VAL_PATH = "D:/smart_tech/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt"


with open(VAL_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)



