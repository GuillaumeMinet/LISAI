from tifffile import imsave

def save_validation_images(list_imgs, folder, volumetric=False):
    """
    list_imgs: list of (x, y, pred)
    """
    for i, (x, y, pred) in enumerate(list_imgs):
        inp = x.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        paired = y is not None
        gt = y.cpu().numpy() if paired else None

        if volumetric:
            inp = inp[0]
            pred = pred[0]
            if paired:
                gt = gt[0]
        elif pred.shape[0] == 1:
            pred = pred[0]

        imsave(folder / f"patch{i:02d}_prediction.tiff", pred, imagej=True)
        imsave(folder / f"patch{i:02d}_input.tiff", inp, imagej=True)
        if paired:
            imsave(folder / f"patch{i:02d}_groundtruth.tiff", gt, imagej=True)
