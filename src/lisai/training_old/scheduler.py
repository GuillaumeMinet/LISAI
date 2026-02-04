import subprocess
import os

# Path to the configuration file and train.py
config_path = os.path.join("src", "training", "configs", "cfg_HDN.py")
train_script = os.path.join("src", "training", "train.py")

# List of training configurations
training_configs = [
    {"exp_name": "HDN_single_GMMsigN2VAvgbis_KL03_noAugm", "training_prm": {"betaKL": 0.3}},
    {"exp_name": "HDN_single_GMMsigN2VAvgbis_KL05_noAugm", "training_prm": {"betaKL": 0.5}},
    {"exp_name": "HDN_single_GMMsigN2VAvgbis_KL07_noAugm", "training_prm": {"betaKL": 0.7}},
]

def modify_config_file(config_path, new_params):
    """Modify the configuration file while preserving formatting and comments."""
    with open(config_path, "r") as file:
        lines = file.readlines()

    updated_lines = []
    exp_name_changed = False
    for line in lines:
        updated_line = line
        for key, value in new_params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        for nested_key, nested_value in sub_value.items():
                            if f'"{nested_key}":' in line:
                                previous_value = line.split(": ")[1].strip().rstrip(",").strip()
                                updated_line = line.replace(
                                    f'"{nested_key}": {previous_value}', f'"{nested_key}": {nested_value}'
                                )
                    elif f'"{sub_key}":' in line:
                        previous_value = line.split(": ")[1].strip().rstrip(",").strip()
                        updated_line = line.replace(
                            f'"{sub_key}": {previous_value}', f'"{sub_key}": {sub_value}'
                        )
            elif f'"{key}":' in line:
                if exp_name_changed and key == "exp_name":
                    continue
                previous_value = line.split(": ")[1].strip().rstrip(",").strip()
                updated_line = line.replace(f'"{key}": {previous_value}', f'"{key}": "{value}"')
                exp_name_changed = True if key == "exp_name" else exp_name_changed
        updated_lines.append(updated_line)

    with open(config_path, "w") as file:
        file.writelines(updated_lines)

def run_training():
    """Run the training script."""
    subprocess.run(["python", train_script], check=True)

# Schedule trainings
for config in training_configs:
    print(f"Starting training: {config['exp_name']}")
    modify_config_file(config_path, config)
    run_training()
    print(f"Finished training: {config['exp_name']}")