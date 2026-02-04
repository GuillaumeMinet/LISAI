import torch


def forward_pass_tiling(x, y, device, model, gaussian_noise_std, patch_size):
    """
    Forward pass for HDN model with tiling. Averages the KL and recon loss
    over all patches.

    Parameters:
    -----------
    x: torch.tensor
        input tensor to feed into model
    y: torch.tensor or None
        ground-truth used for likelihood in case of supervised training.
    device: GPU device
    model: Ladder VAE object
        Hierarchical DivNoising model.
    gaussian_noise_std: float
        std of Gaussian noise used to corrupty data. For intrinsically noisy data, set to None.
    patch_size: int
        size of the tiling patches.

    Returns:
    --------
    output: dict
        dictionnary with the different losses, and the tiled predictions.
    """

    _, _, h, w = x.shape
    
    # Calculate number of patches along height and width
    num_patches_h = (h + patch_size - 1) // patch_size  # Ceiling division
    num_patches_w = (w + patch_size - 1) // patch_size  # Ceiling division
    
    # Prepare empty arrays to accumulate results
    total_recons_loss = 0
    total_kl_loss = 0
    out_mean_tiled = torch.zeros_like(x)  # To accumulate 'out_mean' results
    out_sample_tiled = torch.zeros_like(x)  # To accumulate 'out_sample' results
    total_patches = num_patches_h * num_patches_w

    # Iterate over all patches and call forward_pass for each
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Define the patch slice (taking care of boundary conditions)
            h_start, h_end = i * patch_size, min((i + 1) * patch_size, h)
            w_start, w_end = j * patch_size, min((j + 1) * patch_size, w)

            # Extract patches from x and y
            x_patch = x[:, :, h_start:h_end, w_start:w_end]
            y_patch = y[:, :, h_start:h_end, w_start:w_end] if y is not None else None
            
            # Call forward_pass for the current patch
            patch_output = forward_pass(x_patch, y_patch, device, model, gaussian_noise_std)
            
            # Accumulate losses and outputs
            total_recons_loss += patch_output['recons_loss'] * (x_patch.shape[2] * x_patch.shape[3])
            total_kl_loss += patch_output['kl_loss'] * (x_patch.shape[2] * x_patch.shape[3])
            out_mean_tiled[:, :, h_start:h_end, w_start:w_end] = patch_output['out_mean']
            out_sample_tiled[:, :, h_start:h_end, w_start:w_end] = patch_output['out_sample']
    
    # Average the losses over all patches
    recons_loss = total_recons_loss / (x.shape[2] * x.shape[3])
    kl_loss = total_kl_loss / (x.shape[2] * x.shape[3])
    
    # Prepare output dictionary
    output = {
        'recons_loss': recons_loss,
        'kl_loss': kl_loss,
        'out_mean': out_mean_tiled,
        'out_sample': out_sample_tiled
    }
    return output
    

def forward_pass(x, y, device, model, gaussian_noise_std)-> dict:
    """
    Forward pass for HDN model.

    Parameters:
    -----------
    x: torch.tensor
        input tensor to feed into model
    y: torch.tensor or None
        ground-truth used for likelihood in case of supervised training.
    device: GPU device
    model: Ladder VAE object
        Hierarchical DivNoising model.
    gaussian_noise_std: float
        std of Gaussian noise used to corrupty data. For intrinsically noisy data, set to None.
    
    Returns:
    --------
    output: dict
        dictionnary with the different losses, and the predictions.
    """
    x = x.to(device, non_blocking=True)
    if y is not None:
        y = y.to(device, non_blocking=True)
        model_out = model(x,y)
    else:
        model_out = model(x)
    if model.mode_pred is False:
        
        recons_sep = -model_out['ll']
        kl_sep = model_out['kl_sep']
        kl = model_out['kl']
        kl_loss = model_out['kl_loss']/float(x.shape[2]*x.shape[3])
        
        if gaussian_noise_std is None:
            recons_loss = recons_sep.mean()
        else:
            recons_loss = recons_sep.mean()/ ((gaussian_noise_std/model.data_std)**2)

        
        output = {
                'recons_loss': recons_loss,
                'kl_loss': kl_loss,
                'out_mean': model_out['out_mean'],
                'out_sample': model_out['out_sample']
            }

    else:
        output = {
                'recons_loss': None,
                'kl_loss': None,
                'out_mean': model_out['out_mean'],
                'out_sample': model_out['out_sample']
            }
        
    if 'kl_avg_layerwise' in model_out:
        output['kl_avg_layerwise'] = model_out['kl_avg_layerwise']
        
    return output