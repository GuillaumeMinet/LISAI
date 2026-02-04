CFG = {
    "local": False,
    "mode": 'train', #'train', 'continue_training' or 'retrain'
    "exp_name": "UNetRCAN_Snr-1toAvg_CharEdge",
    
    # Data configuration
    "data_prm": {
        "dataset_name": "Mito_fixed",
        "prep_before":True,
        "already_split": True,
        "canonical_load": True,
        "subfolder": r"preprocess/recon",
        "full_data_path": r"", # used only if not canonical load

        "paired": True,
        "inp": "inp_mltpl_snr",
        "gt": "gt_avg", # for paired datasets only
        
        "mltpl_snr_prm":{
            "snr_idx": -1,
        },


        "initial_crop": None,
        "patch_size": 64,
        "val_patch_size": 256,
        "augmentation": False,
        "patch_thresh": 0.03, # None or float [0,1]
        "select_on_gt": True,
        "filters": ['tiff','tif'],

        "batch_size": 64, # loader batch size (not gpu batch size)

    },

    # Data normalization
    "normalization":{
        "norm_prm": {
            "clip": -3,
        }
    },

    # Model configuration
    "model_architecture": "unet_rcan",
    "model_prm":{
        "UNet_prm": {
            "feat": 64,
            "depth": 3,
            "in_channels": 1,
            "out_channels": 1,
            "activation": "swish",
            "norm": "group",
            "gr_norm": 8,
            "dropout": 0.1,
            "cab_skip_con": False,
        },
        "RCAN_prm": {
            "num_features": 64,
            "num_rg": 6,
            "num_rcab": 8,
            "reduction": 12,
            "dropout": 0.1
        }
    },

    # Training configuration
    "training_prm": {
        "n_epochs": 100,
        "batch_size": 16, # actual gpu batch size
        "lr": 0.00001,
        "val_loss_patience":30,
        "scheduler": "ReduceLROnPlateau",
        "pbar": True,
        "early_stop": False,
    },

    # Loss function configuration
    "loss_function":{
        "name": "CharEdge",
        "MSE_upsampling_prm":{
            "upsampling_factor": 2,
            "alpha": 0.5,
        },

        "CharEdge_loss_prm":{
            "alpha": 0.05,
        },
    },


    # Loading existing model 
    #  => only used in 'continue_training' or 'retrain' modes.
    "load_model": {
        "canonical_load": True,
        "dataset_name": "Vim_fixed_mltplSNR_30nm",
        "subfolder": "Upsampling",
        "exp_name": "Vim_UnetRCAN_bigger_upsampAtTheEnd_Paired_snr1_GTavg_00",
        "model_name": "",
        "model_full_path": r"",
        "load_method" : 'state_dict', #'state_dict' or 'full_model'
    },
    
    # Saving configuration
    "saving_prm": {
        "saving": True, 
        "canonical_save": True,
        "subfolder": r"UNetRCAN",
        "model_full_path": r"",
        "state_dict": True,
        "entire_model": False,
        "overwrite_best": True,
        "save_last": True,
        "config_save_name": "config_train.json",
        "loss_name": "loss.txt",  
        "log_name":  "train_log.log",
        "save_validation_images":True,
        "save_validation_freq": 10, # save every Xth epoch
    },
}