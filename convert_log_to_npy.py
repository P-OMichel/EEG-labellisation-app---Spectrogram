import numpy as np

file = open('anesthesia_database/eeg_producer__09_01_2025__10_41_49__.log','r')
lines = file.readlines()
N = len(lines)
y = []
for i in range(N):
    y.append(float(lines[i]))
y = np.array(y)

np.save('eeg_producer__09_01_2025__10_41_49.npy', y)