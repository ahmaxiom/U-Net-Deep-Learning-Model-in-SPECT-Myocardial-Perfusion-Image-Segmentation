import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pydicom
import os
from scipy import ndimage
from tkinter import Scale, HORIZONTAL

# Create a root window and hide it
root = tk.Tk()
root.title('Adjust Grayscale Levels')

# Create sliders for the minimum and maximum grayscale values
min_val = tk.DoubleVar(value=0)
max_val = tk.DoubleVar(value=255)
min_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, variable=min_val, label='Min Value')
max_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, variable=max_val, label='Max Value')
min_slider.pack()
max_slider.pack()
#root.withdraw()
while True:
    # Open the file dialog and get the file path
    file_path = filedialog.askopenfilename(title="Select dicom image", filetypes=[("dicom files", "*.dcm")])
    image_name = os.path.splitext(os.path.basename(file_path))[0]
    # Use pydicom to read the file
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array
    # Define the ROI drawing function
    roi_points = []
    drawing = False
    def draw_roi(event, x, y, flags, param):
        global roi_points, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                roi_points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi_points.append((x, y))
    # Create a named window and set the mouse callback function
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 500, 500)
    cv2.setMouseCallback('Image', draw_roi)
    minn = np.min(pixel_array)
    maxx = np.max(pixel_array)
    # Iterate over each slice
    for i in range(0, pixel_array.shape[0], 1):
        # Add up four adjacent slices
        slice = np.sum(pixel_array[i:i+1], axis=0)
        slice = ((slice - minn) / (maxx - minn))  
        # Create two lists to store the points of the outer and inner ROIs
        roi_points_outer = []
        roi_points_inner = []
        # Display the slice and let the user draw the ROIs
        for roi_points, color in [(roi_points_outer, (0, 0, 200))]:
            while True:
                slice_copy = slice.copy()
                slice_copy = slice_copy.astype(float)
                # Apply median filter for denoising
                slice_denoised = cv2.medianBlur((slice*255).astype(np.uint8), 3)
                # Apply Bilateral Filtering
                slice_denoised = cv2.bilateralFilter((slice*255).astype(np.uint8), 5, 50, 50)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                #slice_clahe = clahe.apply(slice_denoised)
                # Apply the "Jet" colormap to the adjusted slice
                slice_copy_jet = plt.get_cmap('viridis')(slice_denoised)

                # Convert the colormap from matplotlib's format to OpenCV's format
                slice_copy_jet = (slice_copy_jet * 255).astype('uint8')

                # Convert from RGB to BGR (OpenCV uses BGR instead of RGB)
                slice_copy_jet = cv2.cvtColor(slice_copy_jet, cv2.COLOR_RGB2BGR)
                #slice_copy_jet = cv2.applyColorMap((slice_copy*255).astype(np.uint8), cv2.COLORMAP_JET)
                if len(roi_points) > 0:
                    cv2.polylines(slice_copy_jet, [np.array(roi_points)], False, color, 1)
                # Adjust the grayscale levels based on the slider values
                root.update()
                min_gray = min_val.get()
                max_gray = max_val.get()
                slice_copy_jet = np.clip(slice_copy_jet, min_gray, max_gray)
                slice_copy_jet = (slice_copy_jet - min_gray) / (max_gray - min_gray)
                cv2.imshow('Image', slice_copy_jet)
                key = cv2.waitKey(100) & 0xFF
                # If the 'e' key is pressed, clear the roi_points list
                # If 'c' is pressed, break from the loop
                if key == ord('c'):
                    break
                if key == ord('e'):
                    roi_points.clear()


        # Create a black and white image based on the outer ROI
        if len(roi_points_outer) > 0:
            bw_image_outer = np.zeros_like(slice).astype(np.uint8)  # Start with an array of zeros
            cv2.drawContours(bw_image_outer, [np.array(roi_points_outer)], -1, (255), thickness=cv2.FILLED)

            # Fill the holes in the outer ROI
            bw_image_filled = ((ndimage.binary_fill_holes(bw_image_outer))*255).astype(np.uint8)

            # If the inner ROI was drawn, create a mask for it
            #if len(roi_points_inner) > 0:
            #    mask = np.zeros_like(slice).astype(np.uint8)  # Start with an array of zeros
            #    cv2.drawContours(mask, [np.array(roi_points_inner)], -1, (255), thickness=cv2.FILLED)
            #    # Subtract the mask from the filled image to get the final result
            #    bw_image_final = cv2.subtract(bw_image_filled, mask)
            #else:
            bw_image_final = bw_image_filled        # Save the original slice and the ground truth
        else:
            bw_image_final=np.zeros_like(slice).astype(np.uint8) 
        slice_copy = slice.copy()*255
        slice_copy = slice_copy.astype(np.uint8)
        slice_copy=slice_copy
        cv2.imwrite(f'Original_{image_name}_{i}.TIFF', slice_denoised)
        cv2.imwrite(f'ground_{image_name}_{i}.TIFF',bw_image_final)

        # Clear the ROI points for the next slice
        roi_points.clear()
    key = cv2.waitKey(0)

    # If the ESC key is pressed, exit the loop
    if key == 27:  # 27 is the ASCII value of the ESC key
        break
cv2.destroyAllWindows()
