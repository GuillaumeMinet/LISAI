import numpy as np
from cv2 import INTER_NEAREST, resize

from lisai.data.utils import make_single_4d

_deterministic_mltpl_sampling = {
    2: {
        1: [[0, 0]],
        2: [[0, 0], [1, 1]],
        3: [[0, 0], [0, 1], [1, 1]],
    },
    3: {
        1: [[0, 0]],
        2: [[0, 0], [1, 2]],
        3: [[0, 0], [1, 2], [2, 1]],
        4: [[0, 0], [1, 2], [2, 0], [1, 1]],
        5: [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]],
        6: [[0, 0], [1, 0], [0, 2], [1, 1], [2, 0], [2, 2]],
        7: [[0, 0], [1, 0], [0, 2], [1, 1], [1, 2], [2, 0], [2, 2]],
        8: [[0, 0], [1, 0], [0, 2], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
    },
    4: {
        1: [[0,0]],
        2: [[0,0],[2,2]]
    }
}


def generate_downsamp_inp(inp,downsamp_prm,fix_center=False,context_length = 1,frame_idx=0):
    """
    Generate undersampled input from inp, using parameters from downsampling_prm.

    Arguments:
        - inp: 4D np.array
            input image to be downsampled
        - dowmsamp_prm: dict
            dowmsamplimg parameters dictionary
        - fix_center: bool (default=False):
            to have same sampling at that time point on "middle frame" (for evaluation)
        - context_length: int (default=1)
        - frame_idx: int (default=0)
            ...
    Returns:
        - downsamp_inp: np.array
        - samp_pos: int or None
    """
    
    inp = make_single_4d(inp)

    p = downsamp_prm.get("downsamp_factor",None)
    if p is None or not isinstance(p,int) or p<2:
        raise ValueError (f"arg:`downsamp_factor` should of type:`int` and value > 1,but got {p} instead.")
    
    downsamp_method = downsamp_prm.get("downsamp_method")
    assert downsamp_method is not None, "`downsamp_method` not specified"

    if downsamp_method == "blur":
        dwnsamp_avg = inp.reshape(inp.shape[0],inp.shape[1],
                                  inp.shape[2]//2,2,inp.shape[3]//2,2).mean(axis=(3,5))
        return dwnsamp_avg,None
    
    elif downsamp_method == "multiple":
        assert inp.shape[1] == 1, "downsamp multiple option only available for single frame"
        multiple_prm = downsamp_prm.get("multiple_prm")
        assert multiple_prm is not None
        fill_factor = multiple_prm.get("fill_factor")
        assert fill_factor is not None
        n_ch = int(p**2 * fill_factor)
        downsamp_inp = np.zeros((inp.shape[0],n_ch,int(inp.shape[2]/p),int(inp.shape[3]/p)))
        
        if multiple_prm.get("random",False): # random px selection
            for patch in range(inp.shape[0]):
                idxs = np.random.choice(range(0,p**2), n_ch, replace=False)
                for ch in range(n_ch):
                    y0 = idxs[ch] // p
                    x0 = idxs[ch] % p
                    downsamp_inp[patch,ch] = inp[patch,0,y0::p,x0::p].copy()

        else: # deterministic px selection
            idxs = np.array(_deterministic_mltpl_sampling.get(p).get(n_ch))
            for ch in range(n_ch):
                y0 = idxs[ch,0]
                x0 = idxs[ch,1]
                downsamp_inp[:,ch] = inp[:,0,y0::p,x0::p].copy()
                        
        return downsamp_inp,None

    elif downsamp_method == "random":
        downsamp_inp = np.zeros((inp.shape[0],inp.shape[1],int(inp.shape[2]/p),int(inp.shape[3]/p)))
        for patch in range(inp.shape[0]):
            for ch in range(inp.shape[1]):
                x0 = np.random.randint(0,p)
                y0 = np.random.randint(0,p)
                downsamp_inp[patch,ch] = inp[patch,ch,y0::p,x0::p].copy()
        return downsamp_inp,None

    elif downsamp_method == "real":
        sampling_strategy =  downsamp_prm.get("sampling_strategy",None)
        if sampling_strategy is None:
            return inp[...,1::p,::p].copy(),None 
        else:
            assert type(sampling_strategy) in [list, np.ndarray]
            if isinstance(sampling_strategy,list):
                sampling_strategy = np.array(sampling_strategy)
            assert np.shape(sampling_strategy)[0] == 2
            
            side_frames = int((context_length - 1)/2)

            downsamp_inp = np.zeros(inp.shape[0],inp.shape[1],int(inp.shape[2]/p),int(inp.shape[3]/p))
            frame_idx = 0
            for patch in range(inp.shape[0]):
                if fix_center: 
                     sampling_start_position = (0-side_frames) % (np.shape(sampling_strategy)[1]) #NOTE: assume sampling strategy alway start with the same indexes...
                else:
                    sampling_start_position = (frame_idx-side_frames) % (np.shape(sampling_strategy)[1])
                for frame in range(inp.shape[1]):
                    k = (frame+sampling_start_position) % np.shape(sampling_strategy)[1]
                    y0 = sampling_strategy[0,k]
                    x0 = sampling_strategy[1,k]
                    downsamp_inp[patch,frame] = inp[patch,frame,y0::p,x0::p].clone()
                frame_idx+=1
            return downsamp_inp,None

    elif downsamp_method == "all":
        nInputs = p**2
        downsamp_inp = np.zeros(inp.shape[0],inp.shape[1],nInputs,int(inp.shape[2]/p),int(inp.shape[3]/p))
        for patch in range(inp.shape[0]):
            for frame in range(inp.shape[1]):
                idxInput = 0
                for y in range(p):
                    for x in range(p):
                        downsamp_inp[patch,frame,idxInput] = inp[patch,ch,y::p,x::p].copy()
                        idxInput +=1
        return downsamp_inp,None

    else:
        raise ValueError(f"Downsampling method {downsamp_method} unknown. Can be 'real','blur' or 'random'.")
        



def generate_masked_inp(inp,masking_prm,downsampled_inp = False,downsamp_factor=None):
    """
    Generate a masked input, e.g

    Arguments:
        - inp: 4D np.array
            input image to be downsampled
        - masking_prm: dict
            dowmsamplimg parameters dictionary
        - downsampled_inp: bool, default = False
            if inp is already downsampled, upsampled first to apply mask
        - downsamp_factor: int, default = None
            downsampling factor of the inp.
            mandatory if arg:`downsampled_inp` is True
    Returns:
        - masked_inp: np.array
        - samp_pos: int or None
    """

    inp = make_single_4d(inp)

    if downsampled_inp:
        inp = upscale_inp(inp,downsamp_factor)

    h,w = inp.shape[-2:]
    masked_inp = inp.copy()
    
    mask = masking_prm.get("mask")

    if mask == "random":
        masks=[]
        mask_h, mask_w = masking_prm.get("mask_shape")
        fill_factor = masking_prm.get("fill_factor")
        dynamic = masking_prm.get("dynamic")
        num_pixels = mask_h * mask_w
        num_ones = int(num_pixels * fill_factor)

        if dynamic:
            n = inp.shape[0]*inp.shape[1]
        else:
            n = inp.shape[0]

        for _ in range(n):
            mask = np.concatenate([np.ones(num_ones), np.zeros(num_pixels - num_ones)])
            np.random.shuffle(mask) # Shuffle to randomize
            mask = mask.reshape(mask_h, mask_w).astype(bool)
            masks.append(mask)
        mask = np.stack(masks)

        if dynamic:
            mask = np.reshape(mask,(inp.shape[0],inp.shape[1],mask_h, mask_w)) # [patch,time,mask_h,mask_w]
            tiled_mask = np.tile(mask,(1,1,h//mask_h +1,w//mask_w +1))[...,:h,:w] # [patch,time,h,w]
        else:
            mask = np.expand_dims(mask,axis=1) # [patch,1,mask_h,mask_w]
            tiled_mask = np.tile(mask,(1,masked_inp.shape[1],h//mask_h +1,w//mask_w +1))[...,:h,:w] # [patch,time,h,w]

    else:
        mask = np.asarray(mask,dtype=bool)
        mask_h,mask_w = mask.shape

        assert h % mask_h == 0 and w % mask_w == 0, "mask size should be multiple of patch size"

        if len(mask.shape) == 3:
            raise NotImplementedError #TODO: deal with dynamic masking strategy
        elif len(mask.shape) == 2:  
            tiled_mask = np.tile(mask,(h//mask_h +1, w//mask_w +1))[:h,:w]
            tiled_mask = np.expand_dims(tiled_mask,axis=(0,1))
            tiled_mask = np.repeat(tiled_mask, inp.shape[0], axis=0)
            tiled_mask = np.repeat(tiled_mask, inp.shape[1], axis=1)
    
   
    if masking_prm.get("random_values",False):
        # Create random values in the range of the image's values (same shape as mask)
        random_values = np.random.uniform(np.min(inp), np.max(inp), size=masked_inp.shape)
        masked_inp[~tiled_mask] = random_values[~tiled_mask]
    else:
        masked_inp[~tiled_mask] = 0
        
    return masked_inp


def upscale_inp(inp:np.array,scale:int):
    """
    Make "inp" bigger in the last 2 dimensison, of a factor "scale".
    """
    assert len(inp.shape) == 4, "inp should be 4d: [Patches,T,H,W]"
    sh=inp.shape[0],inp.shape[1],inp.shape[2]*scale,inp.shape[3]*scale
    upscaled_inp = np.empty(shape=sh,dtype=np.float32)

    trgt_sz = (inp.shape[2]*scale,inp.shape[3]*scale)
    for p in range(inp.shape[0]):
        for t in range(inp.shape[1]):
            patch = inp[p,t]
            upscaled_patch = resize(patch,trgt_sz,interpolation = INTER_NEAREST) 
            upscaled_inp[p,t] = upscaled_patch
    return upscaled_inp
