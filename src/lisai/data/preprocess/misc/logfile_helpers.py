import logging
import re
from pathlib import Path
from datetime import datetime

logger = logging.getLogger('Log file helpers')

def init_log(log_file_path, raw_or_rec,pipeline_name, pipeline_prm,mode="new"):
    """
    Create log txt file and initiate with time, script name used
    for preprocessing, and preprocess parameters.
    """
    assert mode in ["new","existing"]
    openmode = "w" if mode == "new" else "a"
    f = open(log_file_path,openmode)
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    if mode == "existing":
        f.write("\n\n")
    f.write(dt_string)
    if mode == "existing":
        f.write("\n----- Updating existing preprocess ----")
    f.write(
        f'\n\nPreprocess {raw_or_rec} data, with pipepeline {pipeline_name}'
        f' and the following parameters:\n {pipeline_prm}\n\n')
    if mode == "new":
        logger.info('Created log file.')
    else:
        logger.info('Updated log file.')


def parse_log_file(log_file_path: Path):
    """
    
    Parses a log file to extract mappings of original file names to new file names.

    The function reads the log file, identifies sections containing file mappings,
    and stores them in a dictionary where keys are original file names and values
    are the corresponding new file names.

    """

    mappings = {}
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    folder_pattern = re.compile(r"Folder: (\w+)")
    file_pattern = re.compile(r"\s+(\S+)\s+(\S+)\s*")
    
    current_folder = None
    for line in lines:
        folder_match = folder_pattern.match(line)
        if folder_match:
            current_folder = folder_match.group(1)
            continue
        
        file_match = file_pattern.match(line)
        if file_match and current_folder is not None:
            original_file_name, new_file_name = file_match.groups()
            mappings[original_file_name] = new_file_name
    
    return mappings
