import cv2
import numpy as np
import argparse
import os

def create_base_image(width: int, height: int) -> np.ndarray:
    """
    Create a white RGBA image with specified dimensions.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers")
    return np.full((height, width, 4), (255, 255, 255, 255), dtype=np.uint8)

def create_circle_mask(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Create a grayscale mask with a centered circle.
    """
    height, width = image.shape[:2]
    if radius <= 0:
        raise ValueError("Radius must be a positive integer")
    if radius * 2 > min(width, height):
        raise ValueError("Radius is too large for the image dimensions")
        
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def create_circle_image(image: np.ndarray, radius: int, alpha: int) -> np.ndarray:
    """
    Create an RGBA image with a red circle and specified alpha.
    """
    if not 0 <= alpha <= 255:
        raise ValueError("Alpha must be between 0 and 255")
        
    height, width = image.shape[:2]
    circle_image = np.zeros((height, width, 4), dtype=np.uint8)
    center = (width // 2, height // 2)
    cv2.circle(circle_image, center, radius, (0, 0, 255, alpha), -1)
    return circle_image

def combine_images(base_image: np.ndarray, circle_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Combine the base image with the circle image using the mask.
    """
    if base_image.shape != circle_image.shape or mask.shape != base_image.shape[:2]:
        raise ValueError("Image and mask dimensions must match")
        
    result = base_image.copy()
    result[mask > 0] = circle_image[mask > 0]
    return result

def save_image(image: np.ndarray, filename: str) -> None:
    """
    Save the image to a file.
    """
    try:
        if not filename.lower().endswith('.png'):
            filename += '.png'
        cv2.imwrite(filename, image)
    except Exception as e:
        raise IOError(f"Failed to save image to {filename}: {str(e)}")

def main(width: int = 100, height: int = 100, radius: int = 40, alpha: int = 128, filename: str = 'red_circle_lower_alpha.png') -> None:
    """
    Main function to create and save an image with a red circle.
    """
    try:
        # Create the base image
        image = create_base_image(width, height)
        
        # Create the circle mask
        mask = create_circle_mask(image, radius)
        
        # Create the red circle image
        circle_image = create_circle_image(image, radius, alpha)
        
        # Combine the images
        final_image = combine_images(image, circle_image, mask)
        
        # Save the result
        save_image(final_image, filename)
        print(f"Image saved successfully as {filename}")
        
    except (ValueError, IOError) as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an image with a red circle.")
    parser.add_argument("--width", type=int, default=100, help="Image width in pixels (default: 100)")
    parser.add_argument("--height", type=int, default=100, help="Image height in pixels (default: 100)")
    parser.add_argument("--radius", type=int, default=40, help="Circle radius in pixels (default: 40)")
    parser.add_argument("--alpha", type=int, default=128, help="Circle alpha (0-255, default: 128)")
    parser.add_argument("--filename", type=str, default="red_circle.png", help="Output filename (default: red_circle_lower_alpha.png)")
    
    args = parser.parse_args()
    main(args.width, args.height, args.radius, args.alpha, args.filename)