'''
File to create json files with annotation mask and add spectrogram data.
'''
import numpy as np
import scipy as sc
import json
from copy import deepcopy
from pathlib import Path
import os


def spectrogram(
    y,
    fs,
    nperseg_factor=1,
    noverlap_factor=0.9,
    nfft_factor=1,
    detrend=False,
    scaling="psd",
    f_cut=45,
):
    nperseg = int(nperseg_factor * fs)
    noverlap = int(noverlap_factor * nperseg)
    nfft = int(nfft_factor * nperseg)
    window = sc.signal.windows.hamming(nperseg, sym=True)

    f_spectro, t_spectro, stft = sc.signal.stft(
        y,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
    )

    Sxx = np.abs(stft) ** 2

    # cut at f_cut
    if len(f_spectro) > 1:
        df = f_spectro[1] - f_spectro[0]
        j = int(f_cut / df)
        j = max(1, min(j, len(f_spectro)))
        f_spectro = f_spectro[:j]
        Sxx = Sxx[:j, :]

    return f_spectro, t_spectro, Sxx


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def enrich_mask_with_spectrograms(
    in_json_path: str | Path,
    out_json_path: str | Path,
    window_list_key: str = "windows",  # where the segments live in your JSON

) -> None:
    
    # --- load json file
    data = load_json(in_json_path)
    # --- load npy recording
    signal_1d = np.load(in_json_path[:-10] + '.npy') # remove the _mask.json part and add the .npy


    out = deepcopy(data)

    windows = out.get(window_list_key, {})
    if not isinstance(windows, dict):
        raise TypeError(f"Expected 'windows' to be a dict, got {type(windows)}")
    
    n = len(signal_1d)

    for win_key, win in windows.items():
        start_s = float(win["window_start_s"])
        end_s = float(win["window_end_s"])
        fs = int(win["fs_hz"])

        start_i = int(round(start_s * fs))
        end_i = int(round(end_s * fs))

        if start_i < 0 or end_i > n or end_i <= start_i:
            raise ValueError(
                f"Invalid indices for window {win_key}: "
                f"start_i={start_i}, end_i={end_i}, n={n}, fs={fs}"
            )

        segment = signal_1d[start_i:end_i]
        
        # NOTE: Normalize all segments
        sqrt_med = np.sqrt(np.median(segment**2))
        factor = 25 / sqrt_med
        # NOTE: uncomment to have only segments of high amplitude normalised
        if factor <= 1:
            segment = segment * factor
        # NOTE: uncomment to normalize all segments
        # segment = segment * factor

        f_spec, t_spec, spec = spectrogram(segment, fs)

        # JSON-serializable: convert numpy arrays to plain lists
        win["f_spec"] = f_spec.tolist()
        win["spectrogram"] = spec.tolist()

    dump_json(out, out_json_path)

#------------------------------------------------------------------------------------------------------------- 
# NOTE: Specify name of folder where annotated mask are | currently mask is only annotations
#------------------------------------------------------------------------------------------------------------- 
in_path_prefix = 'anesthesia_database_Trousseau/' # 'anesthesia_database_Trousseau/'  # 'anesthesia_database/'
elements = os.listdir(in_path_prefix)

for elem in elements:
    if 'mask' in elem:
        in_path = in_path_prefix + elem
        out_path = 'data_mask_spectro/' + elem[:-10] + '_mask_spectro.json'
        # create mask with annotations and add spectrogram
        enrich_mask_with_spectrograms(in_path, out_path)

print('All files have been processed')

in_path_prefix = 'anesthesia_database/' # 'anesthesia_database_Trousseau/'  # 'anesthesia_database/'
elements = os.listdir(in_path_prefix)

for elem in elements:
    if 'mask' in elem:
        in_path = in_path_prefix + elem
        out_path = 'data_mask_spectro/' + elem[:-10] + '_mask_spectro.json'
        # create mask with annotations and add spectrogram
        enrich_mask_with_spectrograms(in_path, out_path)

print('All files have been processed')

# #------------------------------------------------------------------------------------------------------------- 
# # Update dataset
# #------------------------------------------------------------------------------------------------------------- 

# def load_mask_with_spectrograms(path: str | Path) -> dict:
#     path = Path(path)
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)

# # List all elements (files and folders)
# elements = os.listdir('data_mask_spectro/')

# X_eeg =  []
# X_spec = []
# Y = []

# for elem in elements:
#     if '.json' in elem:
#         # load masks and spectrogram
#         data = load_mask_with_spectrograms('data_mask_spectro/' + elem)
#         # load recording (use try except since can be from 2 different folders)
#         try:
#             y = np.load('anesthesia_database/' + data['recording'])
#         except:
#             y = np.load('anesthesia_database_Trousseau/' + data['recording'])
#         # load window key names
#         list_windows_keys = list(data['windows'].keys())
#         # iterate over all windows
#         for i in range(len(data['windows'])):
#             current_window = data['windows'][list_windows_keys[i]]
#             start_s = float(current_window["window_start_s"])
#             end_s = float(current_window["window_end_s"])
#             fs = int(current_window["fs_hz"])

#             start_i = int(round(start_s * fs))
#             end_i = int(round(end_s * fs))

#             signal = y[start_i : end_i]
#             t_signal = np.arange(len(signal)) / fs

#             mask = current_window['mask']

#             t_spec = current_window['t_spec']
#             f_spec = current_window['f_spec']
#             spec = np.array(current_window['spectrogram'])

#             X_eeg.append(signal)
#             X_spec.append(spec)
#             Y.append(mask)


# #--- convert to np arrays
# X_eeg = np.array(X_eeg)
# X_spec = np.array(X_spec)
# Y = np.array(Y)

# np.save('X_Y_dataset/X_eeg', X_eeg)
# np.save('X_Y_dataset/X_spec', X_spec)
# np.save('X_Y_dataset/Y', Y)

# print('Dataset updated')