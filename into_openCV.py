#Install the libraries
#!pip install opencv-python numpy matplotlib (Run in the first cell)

#import and verify
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Print the versions to confirm successful import
print(f"OpenCV Version: {cv2.__version__}")
print(f"NumPy Version: {np.__version__}")

#Simple OpenCV Test (Load & Display an Image)

video_path = 'test_video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read just the first frame
success, frame = cap.read()

# Always release the capture object after you're done
cap.release()

if success:
    print("Successfully read the first frame.")
    
    # --- IMPORTANT ---
    # OpenCV reads images in BGR format (Blue, Green, Red)
    # Matplotlib displays images in RGB format (Red, Green, Blue)
    # We must convert it before displaying it, or the colors will be wrong.
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the image using Matplotlib
    plt.figure(figsize=(10, 6)) # Set a good figure size
    plt.imshow(frame_rgb)
    plt.title("Test Frame from Video")
    plt.axis('off') # Hide the X and Y axes
    plt.show()
    
else:
    print(f"Error: Could not open or read video file at: {video_path}")