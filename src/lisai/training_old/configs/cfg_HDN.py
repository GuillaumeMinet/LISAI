CFG = {
    "local": True,
    "mode": 'train', #'train', 'continue_training' or 'retrain'
    "is_lvae": True,
    "exp_name": "HDN_single_GMMsigN2VAvgbis_KL07_noAugm",

    # Data configuration
    "data_prm": {
        "dataset_name": "Vim_fixed_mltplSNR_30nm",
        "prep_before":True,
        "canonical_load": True,
        "subfolder": r"preprocess/recon",
        "full_data_path": r"", # used only if not canonical load

        "paired": False,
        "already_split": True,
        "inp": "inp_single",
        "gt": None, # for paired datasets 
        
        "initial_crop": None,
        "patch_size": 64,
        "val_patch_size": 256,
        "augmentation": False,
        "patch_thresh": 0.1, # None or float [0,1]
        "select_on_gt": False,
        "filters": ['tiff','tif'],

        "batch_size": 64, 

        # "mltpl_snr_prm":{
        #     "snr_idx": 0,
        # },

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
    "noise_model": "Noise0_SigN2Vavg_Clip-3_norm_bis",

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
        "n_epochs": 100,
        "batch_size": 8,
        "lr": 0.0001,
        "betaKL": 0.7,
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
        "dataset_name": "Vim_20nmPxSize",
        "subfolder": "HDN",
        "exp_name": "HDN_single_GMMsigN2VAvgbis_KL06_40nm",
        "model_name": "",
        "model_full_path": r"",
        "load_method" : 'state_dict', #'state_dict' or 'full_model'
    },
    
    # Saving configuration
    "saving_prm": {
        "saving": True, 
        "canonical_save": True,
        "subfolder": r"HDN",
        "overwrite_best": True,
        "save_last": True,
        "entire_model": True,
        "state_dict": True,
        "save_validation_images":True,
        "save_validation_freq": 10, # save every Xth epoch
    },
}