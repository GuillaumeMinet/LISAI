import numpy as np
import tifffile
from pathlib import Path
def normalize_stack(input_file, output_file):
    # Read the 4D TIFF stack (the shape should be [Z, Y, X, Channels])
    with tifffile.TiffFile(input_file) as tif:
        stack = tif.asarray()

    # Ensure the stack is 4D (Z, Y, X, Channels)
    if stack.ndim != 4:
        raise ValueError("The input TIFF stack must be 4D.")

    # Normalize each 2D image independently
    normalized_stack = np.zeros_like(stack, dtype=np.float32)

    for z in range(stack.shape[0]):  # Iterate through each 2D slice in the stack
        for c in range(stack.shape[1]):  # Iterate through each channel
            # Extract the 2D image (slice) for this Z and channel
            image_2d = stack[z, c]
            normalized_stack[z, c] = (image_2d - np.mean(image_2d))/np.std(image_2d)

    # Save the normalized stack to a new file
    tifffile.imsave(output_file,normalized_stack,imagej=True,metadata={"axes": "TZYX"})

    print(f"Normalized stack saved to {output_file}")


folder = Path(r"\\storage3.ad.scilifelab.se\testalab\Guillaume\01_Projects\DL_monalisa\_paper\Denoising_charac\Model_comp")
input_file = "All_in_one.tif"
output_file = "All_in_one_normalized.tif"
normalize_stack(folder / input_file, folder /output_file)
