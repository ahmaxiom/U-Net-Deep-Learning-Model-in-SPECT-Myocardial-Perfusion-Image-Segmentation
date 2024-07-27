import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
from PIL import Image
import datetime
import os

# Specify your image directory and DICOM file name
image_dir = "C:/Users/admin/Desktop/Spect Final Segmentation Evalaution code/output2"
dicom_filename = "SGATE_G_1001_DS_NEW.dcm"

# Create a new DICOM file
print("Creating new DICOM file...")
file_meta = Dataset()
file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
file_meta.MediaStorageSOPInstanceUID = "1.2.3"
file_meta.ImplementationClassUID = "1.2.3.4"

# Create the FileDataset instance (initially no data elements, but file_meta supplied)
ds = FileDataset(dicom_filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

# Add the data elements
ds.PatientName = "Test^Patient"
ds.PatientID = "123456"

# Set the transfer syntax
ds.is_little_endian = True
ds.is_implicit_VR = True
#ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
ds.is_little_endian = False
ds.is_implicit_VR = False

# Set the date and time
dt = datetime.datetime.now()
ds.ContentDate = dt.strftime('%Y%m%d')
timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
ds.ContentTime = timeStr

images = []
for i in range(480):
    img_path = os.path.join(image_dir, f'mask_{i}.png')
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img_array = np.array(img, dtype=np.uint16)  # Convert to uint16
    images.append(img_array)
pixel_data = np.stack(images)

# Set the pixel data
ds.PixelData = pixel_data.tobytes()

# Set the necessary attributes
ds.Rows = pixel_data.shape[1]
ds.Columns = pixel_data.shape[2]
ds.NumberOfFrames = pixel_data.shape[0]
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.PixelRepresentation = 0
ds.BitsStored = 16
ds.BitsAllocated = 16
ds.HighBit = 15



# Save the DICOM file
print("Saving DICOM file...")
ds.save_as(dicom_filename)
print("DICOM file saved.")
