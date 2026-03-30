'''
File to call when new data is made to create updated dataset. 
Must be called after creating spectrogram mask with create_mask_spectro_file

- compute the features from the {EEG + spectrogram} dataset.
- save as X_feature np.array
'''
import numpy as np
import os
import json
from pathlib import Path
from ML.functions.feature_extractor import extract_features_time_series, extract_features_spectrogram

#================================ DIRECTORIES =============================
data_mask_spectro_dir = 'data_mask_spectro/' # path to where labelled data is stored

name_saved_eeg = 'X_eeg_30_03_2026' # suffix of file name where features are stored
name_saved_spectro = 'X_spec_30_03_2026' # suffix of file name where features are stored
name_saved_mask = 'Y_30_03_2026' # suffix of file name where features are stored
name_saved_features = 'X_features_30_03_2026' # suffix of file name where features are stored

#================================ Functions =============================
def load_mask_with_spectrograms(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def convert_labelled_data_to_dataset(dir):

    elements = os.listdir(dir)
    X_eeg =  []
    X_spec = []
    Y = []

    for elem in elements:
        if '.json' in elem:
            # load masks and spectrogram
            data = load_mask_with_spectrograms('data_mask_spectro/' + elem)
            # load recording (use try except since can be from 2 different folders)
            try:
                y = np.load('anesthesia_database/' + data['recording'])
            except:
                y = np.load('anesthesia_database_Trousseau/' + data['recording'])
            # load window key names
            list_windows_keys = list(data['windows'].keys())
            # iterate over all windows
            for i in range(len(data['windows'])):
                current_window = data['windows'][list_windows_keys[i]]
                start_s = float(current_window["window_start_s"])
                end_s = float(current_window["window_end_s"])
                fs = int(current_window["fs_hz"])

                start_i = int(round(start_s * fs))
                end_i = int(round(end_s * fs))

                signal = y[start_i : end_i]
                t_signal = np.arange(len(signal)) / fs

                # NOTE: Normalize all segments
                sqrt_med = np.sqrt(np.median(signal**2))
                factor = 25 / sqrt_med
                # NOTE: uncomment to have only segments of high amplitude normalised
                if factor <= 1:
                    signal = signal * factor

                mask = current_window['mask']

                t_spec = current_window['t_spec']
                f_spec = current_window['f_spec']
                spec = np.array(current_window['spectrogram'])

                X_eeg.append(signal)
                X_spec.append(spec)
                Y.append(mask)

    #--- convert to np arrays
    X_eeg = np.array(X_eeg)
    X_spec = np.array(X_spec)
    Y = np.array(Y)

    return X_eeg, X_spec, Y, f_spec


# --- SAVE DATASET ---
X_eeg, X_spec, Y, f_spec = convert_labelled_data_to_dataset(data_mask_spectro_dir) 

np.save('X_Y_dataset/' + name_saved_eeg, X_eeg)
np.save('X_Y_dataset/' + name_saved_spectro, X_spec)
np.save('X_Y_dataset/' + name_saved_mask, Y)


# --- COMPUTE FEATURES ---
X = []

fs  = 128
for i in range(len(Y)):
    mean, med_mean, q_std, med_q_std, q_linelen, med_q_linelen, q_env, med_q_env, q_std_env, med_q_std_env  = extract_features_time_series(X_eeg[i], int(fs / 8), Y[i])
    ef, ef_recovery, med_ef, med_ef_recovery, prop_delta, prop_alpha, prop_beta, prop_gamma, q_P_tot, med_q_P_tot, slopes =  extract_features_spectrogram(np.array(f_spec), X_spec[i])
    X.append([mean, med_mean, q_std, med_q_std, q_linelen, med_q_linelen, q_env, med_q_env, q_std_env, med_q_std_env, ef, ef_recovery, med_ef, med_ef_recovery, prop_delta, prop_alpha, prop_beta, prop_gamma, q_P_tot, med_q_P_tot, slopes])

# --- SAVE FEATURES DATASET ---
np.save('X_Y_dataset/' + name_saved_features, X)