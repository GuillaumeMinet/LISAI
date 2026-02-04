from tifffile import imread, imwrite
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

def enhance_contrast(image, saturated_percent=0.35):
    """
    Enhance contrast by saturating a percentage of pixels at low and high ends,
    then rescaling intensities linearly between these clipped bounds.
    
    Parameters:
    - image: np.ndarray, input image (any dtype)
    - saturated_percent: float, percentage of pixels to saturate (e.g., 0.35 for 0.35%)
    
    Returns:
    - np.ndarray, contrast enhanced image with same dtype as input
    """
    low_percentile = saturated_percent / 2
    high_percentile = 100 - low_percentile
    
    low_val = np.percentile(image, low_percentile)
    high_val = np.percentile(image, high_percentile)
    
    # Clip and scale
    clipped = np.clip(image, low_val, high_val)
    scaled = (clipped - low_val) / (high_val - low_val)
    
    # Rescale back to original dtype range
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        scaled = (scaled * info.max).astype(image.dtype)
    else:
        # For float images, keep as float between 0 and 1
        scaled = scaled.astype(image.dtype)
    
    return scaled


def add_colorbar(image_stack, zmax=0,bar_fraction=0.25, bar_height=20,position="bottom_right",
                 bar_text_dist=3,border_margin=3,colormap="turbo"):
    """
    Adds a Turbo colormap colorbar with min/max Z labels on specified corner of each RGB frame.
    Ensures the colorbar never overlays the text by adjusting bar width and placement.
    Keeps original image dimensions (no resizing).
    
    Parameters:
    - image_stack: np.ndarray (T, Y, X, 3), uint8 RGB images
    - zmax: float, maximum Z value for the colorbar. If 0, labels not added (default 0)
    - bar_fraction: float, fraction of image width for colorbar + text (default 0.25)
    - bar_height: int, height of colorbar in pixels (default 20)
    - position: str, one of "bottom_left", "bottom_right", "top_left", "top_right"
    - bar_text_dist: int, distance between colorbar and text labels in pixels (default 3)
    - border_margin: int, margin from image edge to colorbar in pixels (default 3)
    - colormap: str, name of the colormap to use (default "turbo")
    
    Returns:
    - np.ndarray same shape as image_stack with overlay added
    """
    T, Y, X, C = image_stack.shape
    assert C == 3, "Input image stack must be RGB"
    assert position in ("bottom_left", "bottom_right", "top_left", "top_right"), "Invalid position"
    
    if zmax!=0:
        z0_label = "0 µm"
        zmax_label = f"{zmax:.1f} µm"

        # Setup font
        font_size = int(bar_height * 0.8)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
        # Create dummy draw context for text measurement
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        z0_bbox = draw.textbbox((0, 0), z0_label, font=font)
        z0_w = z0_bbox[2] - z0_bbox[0]
        text_height = z0_bbox[3] - z0_bbox[1]

        zmax_bbox = draw.textbbox((0, 0), zmax_label, font=font)
        zmax_w = zmax_bbox[2] - zmax_bbox[0]

        # Total horizontal padding needed for text + small gap on each side
        bar_pad = z0_w + zmax_w + bar_text_dist * 2
    else:
        z0_label = ""
        zmax_label = ""
        z0_w = 0
        zmax_w = 0
        text_height = 0
        bar_pad = 0 
    
    # Calculate available width for colorbar = total fraction of image width minus padding on both sides
    total_bar_area = int(X * bar_fraction)
    colorbar_width = max(total_bar_area - bar_pad, 10)  # minimum width 10px
    
    # Generate colorbar
    try:
        colorbar = (colormaps[colormap](np.linspace(0, 1, colorbar_width))[:, :3] * 255).astype(np.uint8)
    except KeyError:
        raise ValueError(f"Colormap '{colormap}' not found. Available colormaps: {list(colormaps.keys())}")
    colorbar = np.tile(colorbar[np.newaxis, :, :], (bar_height, 1, 1))  # (bar_height, colorbar_width, 3)
    
    out = image_stack.copy()
    
    # Calculate positions
    if "bottom" in position:
        bar_y = Y - bar_height - border_margin
        text_y_pos = bar_y + (bar_height - text_height) // 2
    else:  # top
        bar_y = border_margin
        text_y_pos = border_margin + (bar_height - text_height) // 2
    
    if "left" in position:
        z0_text_x = border_margin
        bar_x = z0_text_x + z0_w + bar_text_dist
        zmax_text_x = bar_x + colorbar_width + bar_text_dist
    else:  # right
        zmax_text_x = X - border_margin - zmax_w
        bar_x = zmax_text_x - colorbar_width - bar_text_dist
        z0_text_x = bar_x - z0_w - bar_text_dist
    
    for t in range(T):
        # Overlay colorbar
        out[t, bar_y:bar_y+bar_height, bar_x:bar_x+colorbar_width, :] = colorbar
        
        pil_img = Image.fromarray(out[t])
        # Draw text labels
        if zmax!=0:
            draw = ImageDraw.Draw(pil_img)
            draw.text((z0_text_x, text_y_pos), z0_label, font=font, fill=(255, 255, 255))
            draw.text((zmax_text_x, text_y_pos), zmax_label, font=font, fill=(255, 255, 255))
        
        out[t] = np.array(pil_img)
    
    return out


def create_color_coded_image(im,colormap="turbo",stack_order="TZYX"):
    """
    Create color-coded images with a Turbo colormap and save them as multipage TIFFs.
    
    Parameters:
    - im: np.ndarray, input image stack
    - colormap: str, name of the colormap to use (default "turbo")
    - stack_order: str, order of dimensions in the input image (default "TZYX")

    Returns:
    - np.ndarray (T, Y, X, 3), color-coded RGB image stack

    """
    if stack_order == "ZTYX":
        im = np.transpose(im, (1, 0, 2, 3))
    elif stack_order != "TZYX":
        raise ValueError(f"Unsupported stack order '{stack_order}'. Use 'TZYX' or 'ZTYX'.")
    
    T, Z, Y, X = im.shape
    # print(f"Input shape: {im.shape}")
    output_rgb = np.zeros((T, Y, X, 3), dtype=np.float32)
    try:
        colors = colormaps[colormap](np.linspace(0, 1, Z))[:, :3]  # Zx3
    except KeyError:
        raise ValueError(f"Colormap '{colormap}' not found. Available colormaps: {list(colormaps.keys())}")
    
    im_norm = np.zeros_like(im, dtype=np.float32)
    for t in range(T):
        frame = im[t]
        norm = (frame - frame.min()) / (frame.max() - frame.min())
        im_norm[t] = norm

    for z in range(Z):
        for c in range(3):
            output_rgb[:, :, :, c] += im_norm[:, z] * colors[z, c]

    # print(f"Output shape: {output_rgb.shape}")

    # Normalize to [0, 255] and convert
    output_rgb = (output_rgb-np.min(output_rgb)) / (np.max(output_rgb) - np.min(output_rgb))
    output_uint8 = (output_rgb * 255).astype(np.uint8)

    return output_uint8


if __name__ == "__main__":
    cmap="rainbow"
    path = r"E:\dl_monalisa\Data\Mito_fast\20250326\aa"
    path = Path(path)
    filters = ['tiff', 'tif']
    list_files = [f for f in os.listdir(path) if f.lower().endswith(tuple(filters))]
    n_files = len(list_files)
    for i,file in enumerate(list_files):
        print(f"Processing #{i}/{n_files}")
        im = imread(path / file)
        if im.ndim < 4:
            print(f"Skipping {file} - not a 4D image")
            continue
        # im = im[:,2:-2]
        # print(im.shape)
        output_uint8 = create_color_coded_image(im, colormap=cmap,stack_order="ZTYX")
        output_uint8 = enhance_contrast(output_uint8,saturated_percent=0.25)

        # Create a Turbo colorbar image (e.g., vertical)
        zmax = (im.shape[0]-1) * 0.4
        with_bar = add_colorbar(output_uint8, zmax=zmax, bar_fraction=0.25, bar_height=20,position="bottom_right",colormap=cmap)

        # Save as multipage RGB TIFF
        imwrite(path / f"{file}_colorCoded.tif", output_uint8, photometric='rgb')
        imwrite(path / f"{file}_colorCoded_withColorBar.tif", with_bar, photometric='rgb')
        # break
        