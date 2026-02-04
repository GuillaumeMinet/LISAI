CFG = {
    "local": True,
    "mode": 'train', #'train', 'continue_training' or 'retrain'
    "is_lvae": True,
    "exp_name": "Snr7_supervisedAvg_betaKL1e-5",

    # Data configuration
    "data_prm": {
        "dataset_name": "Mito_fixed",
        "prep_before":True,
        "canonical_load": True,
        "subfolder": r"preprocess/recon",
        "full_data_path": r"", # used only if not canonical load

        "paired": True,
        "already_split": True,
        "inp": "inp_mltpl_snr",
        "gt": "gt_avg", # for paired datasets 
        
        "initial_crop": None,
        "patch_size": 64,
        "val_patch_size": 256,
        "augmentation": False,
        "patch_thresh": 0.05, # None or float [0,1]
        "select_on_gt": True,
        "filters": ['tiff','tif'],

        "batch_size": 64, 

        "mltpl_snr_prm":{
            "snr_idx": 7,
        },

        # "downsampling":{
        #     "supervised_training": False,
        #     "downsamp_factor": 2,
        #     "downsamp_method": "real",
        # }
    },
    
    # Normalization parameters (automatic load from noise model, or specify in "norm_prm")
    "normalization":{
        "load_from_noise_model": True,
        "norm_prm": None
    },

    # Noise Model
    "noise_model": "Mito_fixed_Noise14914_SigSnrAvg_Clip-3_norm",

    # Model configuration
    "model_architecture": "lvae",
    "model_prm": {
        "num_latents": 5,
        "z_dims": 32,
        "blocks_per_layer": 6,
        "batchnorm": True,
        "free_bits": 1.0,
    },

    # Training configuration
    "training_prm": {
        "n_epochs": 500,
        "batch_size": 32,
        "lr": 0.000005,
        "betaKL": 0.00001,
        "val_loss_patience":30,
        "scheduler": "ReduceLROnPlateau",
        "pbar": True,
        "early_stop": False,
    },

    # Loading existing model 
    #  => only used in 'continue_training' or 'retrain' modes.
    #  => if canonical_load: uses config_project.py structure
    #  => otherwise uses model_path
    "load_model": {
        "canonical_load": True,
        "dataset_name": "Mito_fixed",
        "subfolder": "HDN",
        "exp_name": "Snr9_supervisedSnr0_betaKL1e-2",
        "model_name": None,
        "model_full_path": r"",
        "load_method" : 'state_dict', #'state_dict' or 'full_model'
    },
    
    # Saving configuration
    "saving_prm": {
        "saving": True, 
        "canonical_save": True,
        "subfolder": r"HDN",
        "overwrite_best": False,
        "save_last": True,
        "entire_model": True,
        "state_dict": True,
        "save_validation_images":True,
        "save_validation_freq": 10, # save every Xth epoch
    },
}