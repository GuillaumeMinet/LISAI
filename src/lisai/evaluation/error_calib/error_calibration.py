import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
from careamics.lvae_training.calibration import (
    Calibration,
    plot_calibration,
)
from pathlib import Path

mmse_limit = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tifffile import imread, imsave
import os
import numpy as np

model_path = Path(r"E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Denoising_selected\HDN_single_GMMsigN2VAvgbis_KL05_noAugm_02")
val_images_path = model_path / "evaluation_last_train"
test_images_path = model_path / "evaluation_last"

eps = 0.01
n_files = 3

# test images loading
pred_test = []
std_test = []
target_test = []

for i in range(n_files):
    mmse_pred = imread(test_images_path / f"img_{i}_pred.tif")
    image_gt = imread(test_images_path / f"img_{i}_gt.tif")
    samples = imread(test_images_path / f"img_{i}_samples.tif")

    min_value = np.min([np.min(samples),np.min(mmse_pred)])

    mmse_pred = mmse_pred - min_value + eps
    samples = samples - min_value + eps
    
    std_pixel = np.std(samples,axis=0,keepdims=True)

    mmse_pred = np.expand_dims(mmse_pred,axis=(0,-1))
    image_gt = np.expand_dims(image_gt,axis=(0,-1))
    std_pixel = np.expand_dims(std_pixel,axis=(-1))
    
    pred_test.append(mmse_pred)
    std_test.append(std_pixel)
    target_test.append(image_gt)

pred_test = np.concatenate(pred_test, axis=0)[..., np.newaxis]
std_test = np.concatenate(std_test, axis=0)[..., np.newaxis]
target_test = np.concatenate(target_test, axis=0)[..., np.newaxis]

# val images loading
pred_val = []
std_val = []
target_val = []
for i in range(5):
    mmse_pred = imread(val_images_path / f"img_{i}_pred.tif")
    image_gt = imread(val_images_path / f"img_{i}_gt.tif")
    
    samples = imread(val_images_path / f"img_{i}_samples.tif")
    std_pixel = np.std(samples,axis=0,keepdims=True)

    mmse_pred = np.expand_dims(mmse_pred,axis=(0,-1))
    image_gt = np.expand_dims(image_gt,axis=(0,-1))
    std_pixel = np.expand_dims(std_pixel,axis=(-1))
    
    pred_val.append(mmse_pred)
    std_val.append(std_pixel)
    target_val.append(image_gt)

pred_val = np.concatenate(pred_val, axis=0)[..., np.newaxis]
std_val = np.concatenate(std_val, axis=0)[..., np.newaxis]
target_val = np.concatenate(target_val, axis=0)[..., np.newaxis]



### CALIBRATION ###

calib = Calibration(
    num_bins=50,
)
native_stats = calib.compute_stats(
    pred=pred_val,
    pred_std=std_val,
    target=target_val
)
count = np.array(native_stats[0]['bin_count'])
count = count / count.sum()


# Compute calibration factors for the channels
calib_factors, factors_array = calib.get_calibrated_factor_for_stdev(pred_val, std_val, target_val)


# Use calibration factor we previously computed on the Validation data...

num_bins = 50
show_identity = True

# ...on the validation data
print('Compute calibration for validation data...',end='')
calib_val = Calibration(num_bins=num_bins)
stats_val = calib_val.compute_stats(
    pred_val,
    std_val * factors_array["scalar"] + factors_array["offset"],
    target_val
)
print('✅')

# ...on the test data
print('Compute calibration for test data...',end='')
calib_test = Calibration(num_bins=num_bins)
stats_test = calib_test.compute_stats(
    pred_test,
    std_test * factors_array["scalar"] + factors_array["offset"],
    target_test
)
print('✅')

stats_test_unscaled = calib_test.compute_stats(
    pred_test,
    std_test,
    target_test
)


# Finally, plotting the results!
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Test-data Calibration (Scaled)")
plot_calibration(ax, stats_test, None, show_identity=show_identity, scaling_factor=factors_array["scalar"].item(), offset=factors_array["offset"].item())
plt.savefig(model_path / 'Calibration_Scaled.png', dpi=300, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Test-data Calibration (Scaled-Unscaled)")
plot_calibration(ax, stats_test, stats_test_unscaled, show_identity=show_identity, scaling_factor=factors_array["scalar"].item(), offset=factors_array["offset"].item())
plt.savefig(model_path / 'Calibration_Scaled.png_Unscaled.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Compute the predicted true error
pred_error_test = std_test * factors_array["scalar"].squeeze() + factors_array["offset"].squeeze()
pred_error_test_normalized = pred_error_test / pred_test
_, ax = plt.subplots(figsize=(15, 10), ncols=3, nrows=1)
hs = 0
ws = 0
sz = pred_test.shape[-3]
rand_sample = False
if rand_sample:
    rand_sample = np.random.randint(0, pred_test.shape[0])
else:
    rand_sample = 0


diff_pred_target   = np.abs(pred_test - target_test)**2
true_L2_error      = np.sqrt(np.mean(diff_pred_target))
estimated_L2_error = np.mean(pred_error_test)

save_path = model_path / "Diff_pred_target.tif"
imsave(save_path,diff_pred_target)
save_path = model_path / "Calibrated_error.tif"
imsave(save_path,pred_error_test)
save_path = model_path / "Calibrated_error_normalized.tif"
imsave(save_path,pred_error_test_normalized)


ax[0].imshow(pred_test[rand_sample, hs:hs + sz, ws:ws + sz, 0])
ax[1].imshow(pred_error_test[rand_sample, hs:hs + sz, ws:ws + sz, 0], cmap='coolwarm')
ax[2].imshow(target_test[rand_sample, hs:hs + sz, ws:ws + sz, 0])

ax[0].set_title('Prediction')
ax[1].set_title('Estimated RMSE')
ax[2].set_title('Target')

# Add text below the center plot (ax[1])
metrics_text = f'True L2 Error: {true_L2_error:.4f} | Est. L2 Error: {estimated_L2_error:.4f} | Diff: {np.abs(estimated_L2_error-true_L2_error):.4f}'

ax[1].text(0.5, -0.1, metrics_text,
           horizontalalignment='center',
           verticalalignment='top',
           transform=ax[1].transAxes,
           fontsize=10)

plt.savefig(model_path / 'Calibration_Error.png', dpi=300, bbox_inches='tight')
plt.show()