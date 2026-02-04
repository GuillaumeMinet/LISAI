import numpy as np
import random
from typing import Union
from tifffile import imsave
import math
directionList = ["h+","h-","v+","v-","h+v+","h+v-","h-v+","h-v-"]



def apply_movement(data,prm,volumetric=False):
    paired = False
    if isinstance(data,tuple):
        inps,gts = data
        if gts is not None:
            paired = True
    else:
        inps = data

    assert len(inps.shape) == 4, "expecting data to be 4d"

    if inps.shape[1] > 1:
        prm["nFrames"] = inps.shape[1]
    else:
        assert prm.get("nFrames",None) is not None
    
    movement_type = prm.get("movement_type")
    if movement_type == "translation":
        moving_inps = []
        moving_gts = [] if paired else None
        for i in range (inps.shape[0]):
            
            if paired:
                _inp,_gt = translation([inps[i],gts[i]],**prm)
                if not volumetric: #single frame target
                    idx=prm.get("nFrames")//2
                    _gt = _gt[idx:idx+1] 

                moving_inps.append(_inp)
                moving_gts.append(_gt)

            else:
                _inp = translation(inps[i],**prm)
                moving_inps.append(_inp)
        
        moving_inps = np.stack(moving_inps,axis=0)
        if paired:
            moving_gts = np.stack(moving_gts,axis=0)
    
    else:
        raise ValueError(f"Movement {movement_type} unknown")

    return moving_inps,moving_gts


    

def translation(imgs: Union[np.ndarray,list],speed:float,direction:str,
                nFrames:int = None,dynamic_direction=False,variable_speed=False,
                keep_center_fixed = False,keep_input_size = False,**kwargs):
    """
    Parameters
        imgs: np.ndarray or list of np.ndarray
            If sequence,same artificial translating motion will be applied for all arrays (e.g. inp,gt)
            if 2D array(s), movement is created by repeating the same frame, times `nFrames`
            if 3D array(s), movement is just added to each already existing frame.
        speed: float or int
            speed of the movement in px/frame. If variable_speed, corresponds to the maximal speed allowed.
        direction: str
            direction of the translation. Should exist in "directionList".
            if "random": one from directionList is chosen randomly either for the entire stack,
            or dynamically for each frame, depending on arg:`dynamic_direction`
        nFrames(opt): int, default = None
            number of frames of output. Only used if imgs are 2D arrays.
        dynamic_direction (opt): bool, default = False
            sets if movement direction is constant or not, ONLY IF arg:`direction` is "random".
        variable_speed (opt): bool, default = False
            sets if movement speed is constant or not. If variable, arg:`speed` sets the maximal speed.
        keep_center_fixed (opt): bool, default = False
            to keep the center frame unmoved
        
    Returns:
        movingStacks: numpy arrays or a list of numpy arrays
    """

    if isinstance(imgs,np.ndarray):
        imgs = [imgs]

    shape = imgs[0].shape
    assert len(shape) in [2,3], "expecting image to be 2d or 3d"
    h,w = shape[-2:]

    if len(shape) == 2 or shape[0]==1:
        assert nFrames is not None
        repeatFrame = True
    else:
        repeatFrame = False
        nFrames = shape[0]

    if direction == "random":
        idx = np.random.randint(0,len(directionList))
        direction = directionList[idx]
        # print(direction)
    else:
        assert direction in directionList, f"unkown direction {direction}"

    maxDist = np.max([round(nFrames * speed),1])
    assert maxDist < 0.1*w or maxDist < 0.1*h, "movement seems to be bigger than a tenth of image size"
    movingStacks = [np.empty(shape=(nFrames,h+2*maxDist+2,w+2*maxDist+2)) for i in range(len(imgs))]
    middle = nFrames // 2
    
    # first calculate all distances 
    distances = np.empty(shape=(2,nFrames),dtype=np.int8)
    distances[0,0] = 0
    distances[1,0] = 0
    distx=0
    disty=0
    for i in range(1,nFrames):
        if dynamic_direction:
            idx = np.random.randint(0,len(directionList))
            direction = directionList[idx]
        if variable_speed:
            step = random.uniform(0,1)*speed
        else:
            step = speed

        if "h+" in direction:
            distx = distances[1,i-1] + step
        if "h-" in direction:
            distx = distances[1,i-1] - step
        if "v+" in direction:
            disty = distances[0,i-1] + step
        if "v-" in direction:
            disty = distances[0,i-1] -step

        if i == middle:
            middle_dist = np.array(([[disty],[distx]]))
        
        distances[0,i] = disty
        distances[1,i] = distx

    # print(distances)
    # optional distance adjustement to keep center frame fixed
    if keep_center_fixed:
        distances = distances - middle_dist
    # print(distances)
    
    # apply move to every frames
    for i in range(nFrames):
        disty = distances[0,i]
        distx = distances[1,i]
        for idx,im in enumerate(imgs):
            movingStack = movingStacks[idx]
            if repeatFrame:
                movingStack[i,(maxDist+1)+round(disty):-(maxDist+1)+round(disty),(maxDist+1)+round(distx):-(maxDist+1)+round(distx)]=im
                # if i == 0 or i == middle or i==nFrames-1:
                #     movingStack[i,(maxDist+1)+round(disty):-(maxDist+1)+round(disty),(maxDist+1)+round(distx):-(maxDist+1)+round(distx)]=im
                # else:
                #     movingStack[i,(maxDist+1)+round(disty):-(maxDist+1)+round(disty),(maxDist+1)+round(distx):-(maxDist+1)+round(distx)]=np.zeros_like(im)
            else:
                movingStack[i,(maxDist+1)+round(disty):-(maxDist+1)+round(disty),(maxDist+1)+round(distx):-(maxDist+1)+round(distx)]=im[i]

    
    for i,movingStack in enumerate(movingStacks):
        movingStacks[i] = movingStack[:,2*maxDist:-2*maxDist,2*maxDist:-2*maxDist]
        if keep_input_size:
            padh = (math.floor((h - movingStacks[i].shape[1]) / 2), math.ceil((h - movingStacks[i].shape[1]) / 2))
            padw = (math.floor((w - movingStacks[i].shape[2]) / 2), math.ceil((w - movingStacks[i].shape[2]) / 2))
            movingStacks[i] = np.pad(movingStacks[i],pad_width=((0,0),padh,padw),mode='constant', constant_values=0)

    if len(movingStacks) == 1:
        return movingStacks[0]
    else:
        # imsave("inp.tif",movingStacks[0])
        # imsave("gt.tif",movingStacks[1])
        return movingStacks





# testing
if __name__ == "__main__":
    from tifffile import imread,imsave
    im1 = imread(r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\preprocess\recon\gt_avg\train\c00.tif")
    # im2 = imread(r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\preprocess\recon\gt_avg\train\c02.tif")
    

    prm = { 
        "artifical_movement":{
            "movement_type": "translation",
            "translation_prm":{
                "keep_center_fixed": False,
                "speed": 10,
                "direction": "h+v+",
                "nFrames": 5,
                "dynamic_direction": False,
                "variable_speed": False,
                "keep_input_size": True,
                }
            }
        }

    artifical_movement_prm = prm.get("artifical_movement")
    movement_type = artifical_movement_prm.get("movement_type")
    if movement_type == "translation":
        movingImgs = translation([im1],**artifical_movement_prm.get("translation_prm"))
    else:
        raise ValueError(f"Movement type {movement_type} unknown")
    
    imsave("movingIm_keepinputsize_centerNOTfixed.tif",movingImgs)
    # imsave("movingIm2.tif",movingImgs[1])