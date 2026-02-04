CFG = {
    "local": True,
    "mode": 'train', #'train', 'continue_training' or 'retrain'
    "exp_name": "Snr0raw_pairedtoDenoised_mask1001",
    
    # Data configuration
    "data_prm": {
        "dataset_name": "Vim_fixed_mltplSNR_30nm",
        "prep_before":True,
        "already_split": True,
        "canonical_load": True,
        "subfolder": r"preprocess/recon",
        "full_data_path": r"", # used only if not canonical load

        "paired": True,
        "inp": "inp_mltpl_snr",
        "gt": "snr0_denoisedHDN", # for paired datasets only
        
        "mltpl_snr_prm":{
            "snr_idx": 0,
        },

        "initial_crop": None,
        "patch_size": 64,
        "val_patch_size": 512,
        "augmentation": True,
        "patch_thresh": 0.05, # None or float [0,1]
        "select_on_gt": False,
        "filters": ['tiff','tif'],

        "batch_size": 64, # loader batch size (not gpu batch size)

        "masking":{
            "mask": [[1,0],[0,1]]
        }

    },

    # Data normalization
    "normalization":{
        "norm_prm": {
            "clip": -3,
        }
    },

    # Model configuration
    "model_architecture": "unet",
    "model_prm": {
        "feat": 64,
        "depth": 4,
        "in_channels":1,
        "out_channels": 1,
        "activation": "swish", # 'ReLU','swish'
        "norm": 'group', # None, 'group', 'batch'
        "gr_norm": 8,
        "dropout": 0.1
    },

    # Training configuration
    "training_prm": {
        "n_epochs": 200,
        "batch_size": 8, # actual gpu batch size
        "lr": 0.001,
        "val_loss_patience":30,
        "scheduler": "ReduceLROnPlateau",
        "pbar": True,
        "early_stop": False,
    },

    # Loss function configuration
    "loss_function":{
        "name": "MSE",
        "MSE_upsampling_prm":{
            "upsampling_factor": 2,
            "alpha": 0.7,
        },

        "CharEdge_loss_prm":{
            "alpha": 0.1,
        },
    },


    # Loading existing model 
    #  => only used in 'continue_training' or 'retrain' modes.
    "load_model": {
        "canonical_load": True,
        "dataset_name": "Vim_fixed_mltplSNR_30nm",
        "subfolder": "Masking",
        "exp_name": "Unet2d_FixedMask1000_GTavg_MSE",
        "model_name": "",
        "model_full_path": r"",
        "load_method" : 'state_dict', #'state_dict' or 'full_model'
    },
    
    # Saving configuration
    "saving_prm": {
        "saving": True, 
        "canonical_save": True,
        "subfolder": r"Masking",
        "model_full_path": r"",
        "state_dict": True,
        "entire_model": False,
        "overwrite_best": True,
        "save_last": True,
        "save_validation_images":True,
        "save_validation_freq": 10, # save every Xth epoch
    },
}