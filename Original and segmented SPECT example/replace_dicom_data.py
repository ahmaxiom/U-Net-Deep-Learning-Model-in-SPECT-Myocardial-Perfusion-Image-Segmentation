import pydicom

# Load the original DICOM file
original_dcm = pydicom.dcmread("SGATE_G_1001_DS.dcm")

# Load the new DICOM file
new_dcm = pydicom.dcmread("SGATE_G_1001_DS_new.dcm")

# Replace the pixel data in the original file with the pixel data from the new file
original_dcm.PixelData = new_dcm.PixelData

# Save the modified DICOM file
original_dcm.save_as("SGATE_G_1001_DS_N.dcm")
