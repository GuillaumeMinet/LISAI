"""
Combine multiple datasets of already preprocessed data.
In this case, coded for the configuration where there is 
only a gt folder (for upsampling, inp is generated from inp.)
"""

import shutil
import numpy as np
import os
import shutil
from pathlib import Path
from datetime import datetime
from tifffile import imread
import logging
logging.basicConfig(format='%(levelname)s : %(message)s',
                    level='DEBUG')

# where all the datasets are located. 1 folder = 1 dataset
path_data = Path(r"\\monalisa3d\D\DL\dl_monalisa\Data\Vim_live_timelapse_Monalisa1_35nm\dump\recon") 

#extension filters
filters = ['tif','tiff']

# where full combined dataset will be saved
path_combined = path_data / "combined" 

# listing folders only
directories_idx = 1 
list_folders = next(os.walk(path_data))[directories_idx]

#check that combined_datasets folder does not already exist, and if ok, create folders
assert os.path.exists(path_combined) == False, "Combined dataset folder seems to already exists."
os.makedirs(path_combined)

copy_list_oldname = []
copy_list_newname = []
count_files=0
count_frames=0

# to loop over directories only
for folder in list_folders:

    data_dir = path_data / folder

    assert os.path.exists(data_dir) == True, f"Can't find data, for folder {folder}. "

    data_list = sorted(os.listdir(data_dir))

    for i in range(len(data_list)):
        file_name = data_list [i]
        if file_name.split('.')[-1] not in filters:
            logging.warning(f'skipping {file_name} because not in {filters}.')
            continue
        src = data_dir / file_name
        im = imread(src)
        t = im.shape[0] #number of time points
        new_name = f"c{count_files:02d}_t{t:02d}.tif"
        dst = path_combined / new_name
        shutil.copyfile(src,dst)
        copy_list_oldname.append(file_name)
        copy_list_newname.append(new_name)
        logging.info(f"Copied {new_name}")
        count_files += 1
        count_frames += t

logging.info(f'Copied all files. Counted #{count_files} files, #{count_frames} in total.')

# write in log .txt file
log_file_name = "combined_log.txt"
log_file_path = Path(path_data) / log_file_name
if os.path.exists(log_file_path):
    logging.warning(f"{log_file_name} already exists - overwriting it!")

f = open(log_file_path,"w")
logging.info('Writing in log file...')
now = datetime.now() 
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
f.write(dt_string)
f.write('\nCombined dataset from folders: \n')
for folder in list_folders:
    f.write(f'\t- {str(folder)}')    
    f.write('\n')

f.write(f'\nNumber of files: {count_files}.\nTotal number of frames: {count_frames}.')

head = str('original_file_name new_file_name').split()
f.write(f'\n\n {head[0]:<30} {head[1]:<30}')

for i in range (len(copy_list_oldname)):
    old_name = copy_list_oldname[i]
    new_name = copy_list_newname[i]
    f.write(f'\n {old_name:<30} {new_name:<30}')

logging.info('Done.')

f.write ('\n\n')

f.close()






