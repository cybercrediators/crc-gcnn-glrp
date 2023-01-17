import sys
import pandas as pd
import numpy as np

# small script to retrieve samples, labels etc.
if len(sys.argv) != 3:
    print("Usage: grab_stats.py [MAPPED_PREPROCESSED_FILENAME] [LABELS_FILENAME]")
    sys.exit(1)

preprocessed = pd.read_csv(sys.argv[1])
labels = pd.read_csv(sys.argv[2])
assigned_label, counts = np.unique(labels.to_numpy().flatten(), return_counts=True)
label_occurences = dict(zip(assigned_label, counts))

# sample and label stats
print("Sample no.: ", labels.shape[1])
print("Normal/Adenoma samples: ", label_occurences[0])
print("CRC samples: ", label_occurences[1])

# preprocessed stats
print("No. of genes: ", preprocessed.shape[0])
