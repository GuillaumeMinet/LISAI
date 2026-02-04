import sys, os, shutil
import importlib
import logging

import config_preprocess as config
sys.path.append(os.getcwd())
import lisai.config_data as config_data
from lisai.lib.utils.get_paths import get_dataset_path
from split_train_val_test import split
from lisai.data.preprocess.misc.logfile_helpers import init_log


logging.basicConfig(format="%(name)s %(levelname)s: %(message)s",
                    level="INFO")

mode = config.CFG.get("mode","new")
assert mode in ["new","existing"], "mode should be 'new' or 'existing'"

### prepare preprocess folder ###
dataset_dir = get_dataset_path(config.CFG["dataset_name"],local = config.CFG["local"]) 
dump_dir = dataset_dir / config_data.CFG["subfolders"]["dump"]
preprocess_dir= dataset_dir / config_data.CFG["subfolders"]["preprocess"]

logging.info(f"Dataset dump directory : {dataset_dir}")
logging.info(f"Target preprocess directory : {preprocess_dir}")


### Execute recon pipeline ###

if config.CFG["recon"]:

    # import pipeline
    try:
        pipeline = importlib.import_module("pipelines_recon." + config.CFG["recon_pipeline"])
    except ImportError as e:
        logging.critical(f"Recon pipeline not found, stopping execution!"
                         f"\n ImportError: {e}")
        exit()

    # import pipeline parameters
    try:
        pipeline_prm = getattr(config, "CFG_" + config.CFG["recon_pipeline"])
    except AttributeError as e:
        logging.critical(f"Parameters of recon pipeline not found, stopping execution!\n"
                         f"AttributeError: {e}")
        exit()

    # create target directory and deal with overwriting
    trgt_dir = preprocess_dir / "recon" 
    if mode == "new":
        try:
            os.makedirs(trgt_dir)
        except:
            if config.CFG["overwrite"]:
                shutil.rmtree(trgt_dir) # delete existing folder
                os.makedirs(trgt_dir) # create new dir
                logging.warning("Preprocess recon folder already exists,"
                                "and overwrite = True, so overwriting it.")
            else:
                logging.critical("Preprocess recon folder already exists,"
                                "and overwrite = False, so stopping execution.")
                exit()
    elif mode == "existing":
        assert os.path.exists(trgt_dir)

    # init preprocess log file    
    log_file_path = preprocess_dir / "recon_preprocess_log.txt"
    init_log(log_file_path,'recon',config.CFG["recon_pipeline"],pipeline_prm,mode=mode)

    # run pipeline
    pipeline.run(dataset_name = config.CFG["dataset_name"],
                 data_format = config.CFG["data_format"],
                 dump_dir = dump_dir,
                 trgt_dir = trgt_dir,  
                 log_file_path = log_file_path,
                 combine = config.CFG["combine_mltpl_datasets"],
                 filters = config.CFG["filters"]["recon"],
                 **pipeline_prm)
    

### Execute raw pipeline ###    

if config.CFG["raw"]:

    # import pipeline
    try:
        pipeline = importlib.import_module("pipelines_raw." + config.CFG["raw_pipeline"])
    except ImportError as e:
        logging.critical(f"Raw pipeline not found, stopping execution!"
                         f"\n ImportError: {e}")
        exit()

    # import pipeline parameters
    try:
        pipeline_prm = getattr(config, "CFG_" + config.CFG["raw_pipeline"])
    except AttributeError as e:
        logging.critical(f"Parameters of raw pipeline not found, stopping execution!\n"
                         f"AttributeError: {e}")
        exit()

    # create target directory and deal with overwriting
    trgt_dir = preprocess_dir / "raw" 
    try:
        os.makedirs(trgt_dir)
    except:
        if config.CFG["overwrite"]:
            shutil.rmtree(trgt_dir) # delete existing folder
            os.makedirs(trgt_dir) # create new dir
            logging.warning("Preprocess recon folder already exists,"
                             "and overwrite = True, so overwriting it.")
        else:
            logging.critical("Preprocess recon folder already exists,"
                             "and overwrite = False, so stopping execution.")
            exit()
    
    # init preprocess log file    
    log_file_path = preprocess_dir / "raw_preprocess_log.txt"
    init_log(log_file_path,'raw',config.CFG["raw_pipeline"],pipeline_prm)

    # run pipeline
    pipeline.run(dataset_name = config.CFG["dataset_name"],
                 data_format = config.CFG["data_format"],
                 dump_dir = dump_dir,
                 trgt_dir = trgt_dir,  
                 log_file_path = log_file_path,
                 combine = config.CFG["combine_mltpl_datasets"],
                 filters = config.CFG["filters"]["raw"],
                 **pipeline_prm)


### Do the split ###

if config.CFG["split_prm"]["do_split"]:
    split(dataset_name = config.CFG["dataset_name"],
          preprocess_folder = preprocess_dir,
          recon = config.CFG["recon"],
          raw = config.CFG["raw"],
          filters = config.CFG["filters"],
          split_cfg = config.CFG["split_prm"])

                  
