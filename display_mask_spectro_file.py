import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_mask_with_spectrograms(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



data = load_mask_with_spectrograms('data_mask_spectro/rec_20240321_085300_mask_spectro.json')
y = np.load('anesthesia_database/' + data['recording'])
list_windows_keys = list(data['windows'].keys())
print(len(data['windows']))
print(len(list_windows_keys))
# current_window = data['windows'][list_windows_keys[5]]
# start_s = float(current_window["window_start_s"])
# end_s = float(current_window["window_end_s"])
# fs = int(current_window["fs_hz"])

# start_i = int(round(start_s * fs))
# end_i = int(round(end_s * fs))

# signal = y[start_i : end_i]
# t_signal = np.arange(len(signal)) / fs

# mask = current_window['mask']

# t_spec = current_window['t_spec']
# f_spec = current_window['f_spec']
# spec = np.array(current_window['spectrogram'])

# fig, axes = plt.subplots(3, sharex =True, constrained_layout = True)
# axes[0].plot(t_signal, signal)
# axes[1].pcolormesh(t_spec, f_spec, np.log2(spec + 0.0000001), shading = 'nearest', cmap = 'jet', vmin = -4, vmax = 8)
# axes[2].plot(t_spec, mask)

# plt.show()
# print(np.shape(spec))