import cv2
import numpy as np
import os

# Function to do nothing on trackbar change
def nothing(x):
    pass

# Path to the folder containing input images
input_folder = 'input_images/'

# Create a window named trackbars
cv2.namedWindow("Trackbars")

# Create 6 trackbars that will control the lower and upper range of HSV
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# List all image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Iterate through all images in the folder
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    
    # Read the image
    frame = cv2.imread(image_path)
    
    while True:
        # Convert the BGR image to HSV image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get the new values of the trackbar in real time as the user changes them
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        # Set the lower and upper HSV range according to the value selected by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])

        # Filter the image and get the binary mask
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # You can also visualize the real part of the target color (Optional)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert the binary mask to 3 channel image
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Stack the mask, original frame, and the filtered result
        stacked = np.hstack((mask_3, frame, res))

        # Show the stacked frame
        cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))

        # If the user presses ESC, move to the next image
        key = cv2.waitKey(1)
        if key == 27:
            break

        # If the user presses 's', save the HSV values and move to the next image
        if key == ord('s'):
            thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
            print(f"HSV values for {image_file}: {thearray}")

            # Save the HSV values to a file (optional)
            np.save(f'{image_file}_hsv_value', thearray)
            break

# Close all windows when done
cv2.destroyAllWindows()
