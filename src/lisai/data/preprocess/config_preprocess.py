CFG = {
    "local": True,
    "mode": "new", # "new","existing" => existing mode to add to existing preprocess
    "dataset_name": "Mito_fixed",
    "data_format": "mltpl_snr",
    "for_training": True,

    "combine_mltpl_datasets": False, 
    "overwrite": True,
    "recon": True,
    "raw": False,

    "recon_pipeline": "recon_mltpl_snr",

    "raw_pipeline": "",

    "filters":{
        "recon": ['tif','tiff'],
        "raw":['h5','hdf5'],
    },

    "split_prm":{
        "do_split": True,

        "mode": 'random', # 'random', 'reuse', 'manual'

        "mode_parameters":{
            "random":{
                "val_frc": 0.15,
                "test_frc": 0.2,
            },

            "reuse":{
                "dataset_name": "same",
                "subfolder": "recon" 
            },
            "manual":{
                "original_name": False,
                "val": ["c16"],
                "test":["c01","c14","c23"]            
            }
        }    
    }
}


CFG_recon_timelapse_upsamp = {
    "subfolder": "", # subfolder with reconstructed data in dump folder
    "crop_size": None,    # put None if no crop wanted, tuple if not a squared crop
    "clip_neg": False,      
    "gauss_filter": None,    # (sigma,radius)
    "remove_first_frame": True,
    "bleach_correction": False,
}



CFG_recon_mltpl_snr = {
    "subfolder": "", # subfolder with reconstructed data in dump folder
    "gather_frames": False,
    "gather_stridxs":{"start":4,"end":10},
    "gt_types": ["snr0","avg"],
    "first_low_inp": False,
    "registration": False,
    "crop_size": 1300,# None if no crop wanted, tuple if not a squared crop
    "gt_clip_neg": False,      
    "gt_gauss_filter": False,    # (sigma,radius)
    "gt_avg_nFrames": 3,
}



CFG_recon_double_pulse_timelapse = {
    "subfolder_inp" : "recon/Low",
    "subfolder_gt": "recon/High",
    "crop_size": None,
}


CFG_recon_3snr = {
    "subfolder": "", # subfolder with reconstructed data in dump folder
    "list_subfolders": ["high","middle","low"], # NOTE: should start with high so that snr0 is the right one!
    "gt_types": ["snr0","avg"],
    "registration": False,
    "crop_size": 1400,# None if no crop wanted, tuple if not a squared crop
    "gt_clip_neg": False,      
    "gt_gauss_filter": False,    # (sigma,radius)
    "gt_avg_exlude": 1,
}

CFG_recon_single = {
    "subfolder": "selected",
    "crop_size": None,# None if no crop wanted, tuple if not a squared crop
    "clip_neg": False,      
    "gauss_filter": None,    # (sigma,radius)
}

CFG_recon_single_dwnSampGT = {
    "subfolder": "selected",
    "crop_size": 2800,# None if no crop wanted, tuple if not a squared crop
    "clip_neg": False,      
    "gauss_filter": None,    # (sigma,radius)
}

CFG_denoising_existing = {
    "folder_to_apply": "inp_mltpl_snr",
    "new_name_folder": "snr0_denoisedHDN",
    "subfolder": "", # only necessary if folder to apply to is in a subfolder ie .../preprocess/recon/subfolder/folder_to_apply/gt/train
    "stack_selection_idx": 0,
    "splits": ["test","train","val"], 
    "crop_size": None,
    "model_info":{
        "model_dataset":"Vim_fixed_mltplSNR_30nm",
        "model_subfolder": "HDN",
        "model_name": "HDN_snr0_GMMsigAvg_KL05",
        "lvae_num_samples": 50,
    },
}


CFG_simul_lines = {
    "subfolder": "", # only necessary if folder to apply to is in a subfolder ie .../preprocess/recon/subfolder/gt/train
    "pxSizes": [15,20,25,35]
}