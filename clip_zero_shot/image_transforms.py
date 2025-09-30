import numpy as np
from PIL import Image
import cv2

def crop_black_border(img, threshold=10):
    """
    Crops black borders from an X-ray image.
    threshold: pixel intensity below which is considered 'black'
    """
    # Convert to grayscale numpy
    gray = np.array(img.convert("L"))
    
    # Find mask of non-black pixels
    mask = gray > threshold
    
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1   # add 1 since slicing is exclusive
        img = img.crop((x0, y0, x1, y1))
    
    return img

import numpy as np
from PIL import Image

def center_and_pad_square(img, threshold=10, pad_color=(0, 0, 0)):
    """
    Crops black borders, then centers the content in a square canvas with black padding.
    
    Args:
        img: PIL Image
        threshold: pixel intensity below which is considered 'black'
        pad_color: color to use for padding
    """
    # Step 1: Crop black borders
    # Convert to grayscale for border detection
    gray = np.array(img.convert("L"))
    
    # Find mask of non-black pixels
    mask = gray > threshold
    
    if mask.any():
        # Find bounding box of non-black pixels
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        # Crop the image to remove black borders
        img = img.crop((x0, y0, x1, y1))
    
    # Step 2: Center in square with black padding
    width, height = img.size
    max_dim = max(width, height)
    
    # Create square canvas with black background
    # Use same mode as input image
    if img.mode == 'L':  # Grayscale
        square_img = Image.new('L', (max_dim, max_dim), pad_color if isinstance(pad_color, int) else 0)
    else:  # RGB or other
        square_img = Image.new(img.mode, (max_dim, max_dim), pad_color)
    
    # Calculate position to center the image
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    
    # Paste the cropped image centered on the square canvas
    square_img.paste(img, (left, top))
    
    return square_img


def apply_clahe(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    return Image.fromarray(equalized)