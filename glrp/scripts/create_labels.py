import sys
import re
import csv
LABEL_FILE="labels.csv"

if len(sys.argv) != 3:
    print("Usage: create_labels.py [PREPROCESSED_FILE] [LABEL_FILE]")
    sys.exit(1)
PREPROCESSED = sys.argv[1]
LABELS = sys.argv[2]

# get csv labels from preprocessed data (col. headers)
with open(PREPROCESSED, newline='') as f:
    reader = csv.reader(f)
    used_samples = next(reader)
used_samples = used_samples[3:]

# strip to basename
used_samples = [re.split('\_|\.', k)[0] for k in used_samples]
#print(used_samples)

# read assigned label samples
with open(LABELS) as f:
    lines = f.read().splitlines()

# assign found labels from samples
# add them if they exist in the preprocessed data
samples = dict(label.split(" ") for label in lines)
vals = [samples[k] for k in used_samples]
res = dict(zip(used_samples, vals))

# write back csv
with open(LABEL_FILE, 'w') as f:
    w = csv.DictWriter(f, res.keys())
    w.writeheader()
    w.writerow(res)
