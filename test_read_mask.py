import json
import matplotlib.pyplot as plt

with open('anesthesia_database/rec_20240321_085300_mask.json', 'r') as file:
    data = json.load(file)

keys = list(data['windows'].keys())
print(keys)

key = keys[0]

mask = data['windows'][key]['mask']

plt.plot(mask)
plt.show()

