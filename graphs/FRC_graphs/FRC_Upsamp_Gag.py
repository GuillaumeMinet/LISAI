import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphs.utils.eval_folder import get_eval_folder, list_images
from graphs.utils.calculate_metrics import calculate_metrics
from graphs.utils.boxplot import box_plot as new_box_plot
from lisai.data.utils import crop_center, extract_patches

import frc
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
import random
from scipy.ndimage import gaussian_filter

from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# data paths
folders_path = Path(r'E:\lisai\datasets\Gag_timelapses\models\Upsamp')

folder_names = [
    "CL1_Upsamp2_biggerNet_02",
    "CL5_Upsamp2_biggerNet_03",
    "CL1_Upsamp05_biggerNet_01"
]

evaluation_folder = "evaluation_best"
ambiguity_selector = "last_epoch"

cl_list = ["1","2","3"]

patch_size = 1200
smooth_gt = True
show_figure = True
plot_gt = True

# saving parameters
save_figure = True
save_legend = False
save_folder = os.path.join(os.getcwd(), r"src/graphs/saved_graphs")
save_title = "Gag_FRC.svg"
legend_save_title = f"{save_title.split('.')[0]}_legend.svg"

# get file idxs first
min_idx = 1e9
max_idx = 0
for folder in folder_names:
    eval_folder = get_eval_folder(folders_path/folder, evaluation_folder, ambiguity_selector)
    list_files = list_images(eval_folder)
    print(len(list_files))
    if len(list_files) < min_idx:
        min_idx = len(list_files)
    if len(list_files) > max_idx:
        max_idx = len(list_files)

assert min_idx%3==0, "expected to have a folder with inp,gt,pred"
assert max_idx%3==0, "expected to have a folder with inp,gt,pred"

n_min = min_idx // 3
n_max = max_idx // 3

# intialize frc dict
frc_curves = {cl: [] for cl in cl_list}
frc_curves["gt"] = []
avg_frc_curves = {cl: [] for cl in cl_list}
avg_frc_curves["gt"] = []
std_frc_curves = {cl: [] for cl in cl_list}
std_frc_curves["gt"] = []

# gt calculation
eval_folder = folders_path/folder_names[0]/evaluation_folder
eval_folder = get_eval_folder(folders_path/folder_names[0],
                              evaluation_folder,
                              ambiguity_selector)
count = 0
start = int(n_max - n_min)//2
stop = int(start + n_min)
idxs = np.arange(start,stop,1)
for i in idxs:
    print(f"GT {count}/{len(idxs)}")
    count+=1
    gt = imread(eval_folder/f"img_{i}_gt.tif")
    if smooth_gt:
        gt = gaussian_filter(gt,sigma = 0.4,radius = 3)

    gt[gt<-0.3]=-0.3
    gt = (gt - np.mean(gt)) / np.std(gt)
    gt = gt + 0.3
    patches_gt = extract_patches(gt,patch_size)
    for i in range(patches_gt.shape[0]):
        patch_gt = frc.util.apply_tukey(patches_gt[i])
        frc_curve = frc.one_frc(patch_gt)
        if np.isnan(frc_curve).any():
            print(f"Skipping a patch in gt {i} due to NaN values in FRC curve.")
        else:
            frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
            frc_curves["gt"].append(frc_curve)


# prediction FRC
for folder_idx,cl in  enumerate(cl_list):
    count = 0
    eval_folder = get_eval_folder(folders_path/folder_names[folder_idx],
                                  evaluation_folder,
                                  ambiguity_selector)
    n_files = len(list_images(eval_folder)) // 3
    start = (n_files - n_min) // 2
    stop = start + n_min
    idxs = np.arange(start,stop,1)
    print(idxs)
    for i in idxs:
        print(f"Predictino {count}/{len(idxs)} in {cl} context length")
        count+=1
        pred = imread(eval_folder/f"img_{i}_pred.tif")
        pred = pred - np.min(pred)
        patches = extract_patches(pred,patch_size)

        for i in range(patches.shape[0]):
            patch = frc.util.apply_tukey(patches[i])
            patch = (patch - np.mean(patch)) / np.std(patch)
            patch = patch - np.min(patch)
            frc_curve = frc.one_frc(patch)
            if np.isnan(frc_curve).any():
                print(f"Skipping a patch in image {i} due to NaN values in FRC curve.")
            else:
                frc_curve = (frc_curve - np.min(frc_curve)) / (np.max(frc_curve) - np.min(frc_curve))
                frc_curves[cl].append(frc_curve)

frc_curves["gt"] = np.stack(frc_curves["gt"])
avg_frc_curves["gt"] = np.mean(frc_curves["gt"], axis=0)
std_frc_curves["gt"] = np.std(frc_curves["gt"], axis=0)

for cl in cl_list:
    frc_curves[cl] = np.stack(frc_curves[cl])
    avg_frc_curves[cl] = np.mean(frc_curves[cl], axis=0)
    std_frc_curves[cl] = np.std(frc_curves[cl], axis=0)
    
colors_list = ['mediumblue', 'darkred','forestgreen','pink',"cyan"]

# Plot 1frc 
scale = 1/30*1e3
img_size = pred.shape
fig,ax = plt.subplots(1, 1, figsize=(5, 5))
for cl in cl_list:
    frc_curve = avg_frc_curves[cl]
    std_curve = std_frc_curves[cl]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label=cl,linewidth=2,color=colors_list[cl_list.index(cl)])
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.3,color=colors_list[cl_list.index(cl)],
                    linewidth=0)

# gt plot
if plot_gt:
    frc_curve = avg_frc_curves["gt"]
    std_curve = std_frc_curves["gt"]
    xs_pix = np.arange(len(frc_curve)) / img_size[0]
    xs_nm_freq = xs_pix * scale
    ax.plot(xs_nm_freq, frc_curve,label="gt",linewidth=0.5,color="black")
    ax.fill_between(xs_nm_freq, frc_curve - std_curve, frc_curve + std_curve, alpha=0.5)



labels_fontSize=20
ticks_prms={"labelsize":20, "width":2,"length":8}
ticks_positions=[0,5,10,15]
# ax.set_title("1FRC")
ax.set_xlabel("Spatial frequency (µm$^{-1}$)",fontsize=labels_fontSize)
ax.set_ylabel("Correlation",fontsize=labels_fontSize)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.legend()

ax.set_xticks(ticks_positions)
ax.tick_params(axis='x',which='major',**ticks_prms)
ax.tick_params(axis='y',which='major',**ticks_prms)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)


if show_figure:
    plt.show()  

exit()
# Saving
if save_figure:
    fig.savefig(os.path.join(save_folder, save_title), bbox_inches='tight')


save_path = folders_path/ "Upsamp_Mitoch_Metrics_vs_CL.svg"
fig.savefig(save_path, format="svg", bbox_inches='tight')

labels_list = [
    "N=1",
    "N=5"
]

if save_legend:
    custom_lines = [
        Line2D([0], [0], color=colors_list[i], lw=4, solid_capstyle='butt', label=labels_list[i])
        for i in range(len(labels_list))
    ]

    legend_fig = plt.figure(figsize=(2, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    legend = Legend(legend_ax, custom_lines, labels_list, loc='center', frameon=False, fontsize=16, handlelength=1.5, handleheight=1.0)
    legend_ax.add_artist(legend)

    legend_fig.savefig(os.path.join(save_folder, legend_save_title), bbox_inches='tight')
