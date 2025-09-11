import cv2
import numpy as np

# Create a 100x100 image with 4 channels (RGBA)
image = np.full((100, 100, 4), (255, 255, 255, 255), dtype=np.uint8)

# Create a mask (grayscale) for the circle
mask = np.zeros((100, 100), dtype=np.uint8)
center = (50, 50) 
radius = 40 
# Draw a filled circle on the mask
cv2.circle(mask, center, radius, 255, -1)  

circle_image = np.zeros((100, 100, 4), dtype=np.uint8)
# Draw a filled circle with alpha 128
cv2.circle(circle_image, center, radius, (0, 0, 255, 128), -1)

# Apply the circle to the main image using the mask
image[mask > 0] = circle_image[mask > 0]

# Save the final image as a PNG file
cv2.imwrite('red_circle.png', image)