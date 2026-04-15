import cv2
import os

# -----------------------------
# CONFIG
# -----------------------------
input_path = "/home/tondamanati-abhinay/Desktop/capstone/Capstone/DataSet/DS/test/testing image/image.png"

# model input size
TARGET_SIZE = (300, 300)

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread(input_path)

if image is None:
    print("Error: Image not found")
    exit()

# -----------------------------
# RESIZE
# -----------------------------
resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

# -----------------------------
# SAVE (overwrite or new file)
# -----------------------------
folder = os.path.dirname(input_path)
filename = os.path.basename(input_path)

new_name = "resized_" + filename
output_path = os.path.join(folder, new_name)

cv2.imwrite(output_path, resized)

print("Resized image saved at:", output_path)