import yaml
import os

def update_paths(config_path, new_x_eeg_path, new_x_spec_path, new_y_path):
    # Load
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update
    if "x_eeg_path" in config["data"]:
        config["data"]["x_eeg_path"] = new_x_eeg_path
    if "x_spec_path" in config["data"]:
        config["data"]["x_spec_path"] = new_x_spec_path
    config["data"]["y_path"] = new_y_path

    # Save
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated config saved to {config_path}")



in_path_prefix = 'DL/experiments/' 
elements = os.listdir(in_path_prefix)

for elem in elements:
    in_path = in_path_prefix + elem
    update_paths(
        config_path = in_path,
        new_x_eeg_path = "X_Y_dataset/X_eeg_13_04_2026.npy",
        new_x_spec_path = "X_Y_dataset/X_spec_13_04_2026.npy",
        new_y_path = "X_Y_dataset/Y_13_04_2026.npy",
    )
