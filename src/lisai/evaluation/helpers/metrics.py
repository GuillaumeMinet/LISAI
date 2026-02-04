import numpy as np
from pathlib import Path
import os
import logging
from .ra_psnr import RangeInvariantPsnr as ra_psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger("metrics")

_available_metrics = [
    "psnr",
    "ra_psnr",
    "ssim"
]

def calculate_metrics(img_name:str,
                      metrics:list,
                      results: dict,
                      pred:np.array,
                      gt:np.array,
                      inp:np.array = None):
    """
    NOTE: inp not mandatory because in some cases (e.g. upsampling), 
    metrics are not calculated between inp and gt.
    """
    
    results = init_results(results,metrics,inp,img_name)

    # Calculate and add metrics to results
    for metric in metrics:
        if metric not in _available_metrics:
            logger.warning(f"{metric} not available, skipping it.")
            continue
        # print(pred.shape,gt.shape)
        # pred metric 
        value = apply_metric(im=pred,ref=gt,metric_name=metric)
        results = update_results(results,"pred",metric,value)

        # optional  input metric
        if inp is not None:
            value = apply_metric(ref=gt,im=inp,metric_name=metric)
            results = update_results(results,"inp",metric,value)

    return results



def apply_metric(ref,im,metric_name):

    if len(ref.shape)==3:
        ref = ref[0]
        im = im[0]
    if len(im.shape)==4:
        ref = ref[0,0]
        im = im[0,0]

    range = np.max(ref)-np.min(ref)

    if metric_name == "ra_psnr":
        return ra_psnr(ref,im)
    
    else:
        func = locals().get(metric_name)
        if func is None:
            func = globals().get(metric_name)
            if func is None:
                return None
    
        return func(ref,im,data_range=range)


def apply_metric_samples(ref,samples,metric_name):

    list_results = [apply_metric(ref,x,metric_name) for x in samples]
    min_value = min(list_results)
    max_value = max(list_results)

    if metric_name in _higher_better:
        best = max_value
        worst = min_value
    elif metric_name in _lower_better:
        best = min_value
        worst = max_value  
    else:
        raise ValueError(f"{metric_name} not found in 'higher or lower better' lists")
    
    return best, worst


def init_results(results,metrics,inp,img_name):
    
    # init if not existing
    if results is None:
        results = {"imgs": [], "pred":{}}
        for metric in metrics:
            results["pred"][metric]=[]
        if inp is not None:
            results["inp"] = {}
            for metric in metrics:
                results["inp"][metric]=[]
    
    # update imgs with img_name
    imgs = results.get("imgs")
    imgs.append(img_name)
    results["imgs"] = imgs

    return results

def update_results(results,pred_or_inp,metric,value):
    ll = results.get(pred_or_inp).get(metric)
    ll.append(value)
    results[pred_or_inp][metric] = ll
    return results