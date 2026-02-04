from typing import Union
from pathlib import Path
import os,sys
import json

sys.path.append(os.getcwd())
from lisai.config_project import config_data


def update_dataset_json(dataset_name:str,
                        data_type:str,
                        data_format:str = None,
                        structure:dict = None,
                        n_files:int = None,
                        n_frames:int = None,
                        n_snr:Union[int, list] = None,
                        split: str = None,
                        ):
    
    """
    Updates "datasets.json" file with newly preprocessed dataset.
    """

    # Load the existing dataset JSON file
    json_file = Path(os.getcwd()) / config_data.CFG["datasets_json"]
    with open(json_file, 'r') as file:
        dataset_json = json.load(file)

    # If dataset_name does not exist, create it
    if dataset_name not in dataset_json:
        new_dataset = True
        dataset_json[dataset_name] = {
            "data_format": data_format,
            "size": {},
            "split":{},
            "structure": {}
        }
    else:
        new_dataset = False
    
    
    # update size
    if n_files is not None:
        size_entry = {
            "n_files": n_files
        }
        if n_frames is not None:
            size_entry["n_frames"] = n_frames
        if n_snr is not None:
            if isinstance(n_snr,int):
                size_entry["#snr"] = n_snr
            elif isinstance(n_snr,list):
                size_entry["#snr"] = ','.join(str(snr) for snr in n_snr)
        
        dataset_json[dataset_name]["size"][data_type] = size_entry

    # update split
    if split is not None:
        dataset_json[dataset_name]["split"][data_type] = split
    
    # update structure
    if structure is not None:
        if new_dataset:
            dataset_json[dataset_name]["structure"][data_type] = structure
        else:
            curr_structure = dataset_json[dataset_name]["structure"][data_type]
            if structure not in curr_structure:
                dataset_json[dataset_name]["structure"][data_type] = ",".join([curr_structure,structure])

    # Write the updated dataset JSON back to the file
    with open(json_file, 'w') as file:
        json.dump(dataset_json, file, indent=4)