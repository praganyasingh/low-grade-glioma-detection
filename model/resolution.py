from PIL import Image

# Open the TIFF image
img = Image.open("sample.tif")

# Get image resolution in pixels
width, height = img.size
print(f"Image Resolution: {width} x {height} pixels")
