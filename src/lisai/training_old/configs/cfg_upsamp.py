CFG = {
    "local": True,
    "mode": 'train', #'train', 'continue_training' or 'retrain'

    # saving name
    "exp_name": "Avg_unpaired_Mltpl025notRandom_Unet_CharEdge_alpha005",
    
    # Data configuration
    "data_prm": {
        "dataset_name": "Vim_fixed_mltplSNR_30nm",
        "prep_before":True,
        "already_split": True,
        "canonical_load": True,
        "subfolder": r"preprocess/recon",
        "full_data_path": r"", # used only if not canonical load

        "paired": False,
        "inp": "gt_avg",
        "gt": None, # for paired datasets only
        
        # "mltpl_snr_prm":{
        #     "snr_idx": 0,
        # },

        # "gt_transform": {
        #     "gauss_blur": [3,0.65]
        # },

        "initial_crop": None,
        "patch_size": 64,
        "val_patch_size": 256,
        "augmentation": False,
        "patch_thresh": 0.07, # None or float [0,1]
        "select_on_gt": False,
        "filters": ['tiff','tif'],

        "batch_size": 64, # loader batch size (not gpu batch size)

        # "artificial_movement":{
        #     "nFrames":5,
        #     "movement_type": "translation",
        #     "direction": "h+v+",
        #     "speed": 3,
        #     "dynamic_direction": False,
        # },

        "downsampling":{
            "downsamp_factor": 2,
            "downsamp_method": "multiple",
            "multiple_prm":{
                "random": False,
                "fill_factor": 0.25,
            }
        }

    },

    # Data normalization
    "normalization":{
        "norm_prm": {
            "clip": 0,
        }
    },

    # Model configuration
    "model_architecture": "unet",
    # "model_prm":{
    #     "upsamp": 3,
    #     "in_channels": 4,
    #     "out_channels": 1,
    #     "num_features": 64,
    #     "num_rg": 8,
    #     "num_rcab": 12,
    #     "reduction": 16,
    #     "dropout": 0.1
    #     },
    "model_prm": {
        "feat": 64,
        "depth": 4,
        "in_channels": 1,
        "out_channels": 1,
        "activation": "swish",
        "norm": "group",
        "gr_norm": 8,
        "dropout": 0.1,
        "upsampling_factor": 2,
        "upsampling_order": "after",
    },  
    # "model_prm":{
    #     "upsampling_net": "rcan",
    #     "upsampling_factor": 2,
    #     "UNet_prm": {
    #         "feat": 64,
    #         "depth": 3,
    #         "in_channels": 2,
    #         "out_channels":2,
    #         "activation": "swish",
    #         "norm": "group",
    #         "gr_norm": 8,
    #         "dropout": 0.1,
    #         "cab_skip_con": False
    #     },
    #     "RCAN_prm": {
    #         "num_features": 64,
    #         "num_rg": 8,
    #         "num_rcab": 12,
    #         "reduction": 16,
    #         "dropout": 0.1,
    #         "upsamp_kernel_factor":1,
    #     }
    # },

    # Training configuration
    "training_prm": {
        "n_epochs": 100,
        "batch_size": 8, # actual gpu batch size
        "lr": 0.001,
        "val_loss_patience":30,
        "scheduler": "ReduceLROnPlateau",
        "pbar": True,
        "early_stop": False,
    },

    # Loss function configuration
    "loss_function":{
        "name": "CharEdge_loss",
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
        "subfolder": "Upsampling_final",
        "exp_name": "Snr1_pairedAvg_Mltpl05_UnetRCAN_rg8_rcab12_red16_CharEdge_Augm",
        "model_name": "",
        "model_full_path": r"",
        "load_method" : 'state_dict', #'state_dict' or 'full_model'
    },
    
    # Saving configuration
    "saving_prm": {
        "saving": True, 
        "canonical_save": True,
        "subfolder": r"Upsampling_refinement",
        "model_full_path": r"",
        "state_dict": True,
        "entire_model": False,
        "overwrite_best": True,
        "save_last": True,
        "save_validation_images":True,
        "save_validation_freq": 10, # save every Xth epoch
    },
}