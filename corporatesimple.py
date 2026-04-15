import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import requests
import sys

# -----------------------------
# CORPORATE CONFIG (UNCHANGED)
# -----------------------------
panel_power = 400
electricity_price = 5.5
installation_cost_per_kw = 38000
pixel_resolution = 0.280

coverage_factor = 0.75
system_loss = 0.82

# -----------------------------
# CUSTOM FUNCTIONS
# -----------------------------
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_p_bce(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def true_positive_rate(y_true, y_pred):
    y_pred_pos = tf.round(tf.clip_by_value(y_pred,0,1))
    y_true_pos = tf.round(tf.clip_by_value(y_true,0,1))
    return tf.reduce_sum(y_true_pos*y_pred_pos)/(tf.reduce_sum(y_true_pos)+1e-7)

# REQUIRED FOR NEW MODEL
def crop_and_concat_fn(inputs):
    target_tensor, skip_tensor = inputs
    th, tw = tf.shape(target_tensor)[1], tf.shape(target_tensor)[2]
    sh, sw = tf.shape(skip_tensor)[1], tf.shape(skip_tensor)[2]

    dh, dw = sh - th, sw - tw

    skip_tensor = tf.image.crop_to_bounding_box(
        skip_tensor, dh//2, dw//2, th, tw
    )
    return tf.concat([target_tensor, skip_tensor], axis=-1)

# -----------------------------
# LOAD NEW MODEL
# -----------------------------
model = tf.keras.models.load_model(
    "/home/tondamanati-abhinay/Desktop/capstone/Capstone/DataSet/DS/final_unet.keras",
    custom_objects={
        "dice_coef": dice_coef,
        "dice_p_bce": dice_p_bce,
        "true_positive_rate": true_positive_rate,
        "crop_and_concat_fn": crop_and_concat_fn
    }
)

# -----------------------------
# LOAD IMAGE (UNCHANGED)
# -----------------------------
image_path = "/home/tondamanati-abhinay/Desktop/capstone/Capstone/DataSet/DS/train/images/000000000046.jpg"
display_crop = 15

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
#  PREDICTION FIX
# -----------------------------
img_resized = cv2.resize(image_rgb, (300, 300))
img_array = img_resized / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
mask = prediction[0, :, :, 0]

# resize back
mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

# -----------------------------
# BINARY MASK (UNCHANGED)
# -----------------------------
mask_binary = (mask > 0.5).astype(np.uint8)

# -----------------------------
# POLYGON SELECTION (UNCHANGED)
# -----------------------------
points, polygons = [], []

base_img = image[display_crop:-display_crop, display_crop:-display_crop].copy()
display_img = base_img.copy()

def redraw():
    global display_img
    display_img = base_img.copy()

    for i, poly in enumerate(polygons):
        shifted = [(px - display_crop, py - display_crop) for px, py in poly]
        cv2.polylines(display_img, [np.array(shifted)], True, (255,0,0), 2)
        cv2.putText(display_img, f"H{i+1}", shifted[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    shifted_points = [(px - display_crop, py - display_crop) for px, py in points]

    for p in shifted_points:
        cv2.circle(display_img, p, 4, (0,255,0), -1)

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x + display_crop, y + display_crop))
        redraw()

cv2.namedWindow("Select Houses", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Houses", mouse_click)

print("Select ONE house per polygon → Press N to save, Q to finish")

while True:
    cv2.imshow("Select Houses", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("n") and len(points) >= 3:
        polygons.append(points.copy())
        points.clear()
        redraw()

    elif key == ord("r"):
        points.clear()
        redraw()

    elif key == ord("q"):
        if len(points) >= 3:
            polygons.append(points.copy())
        break

cv2.destroyAllWindows()

# -----------------------------
# LAT/LON FIX (SAFE)
# -----------------------------
if len(sys.argv) >= 3:
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
else:
    lat, lon = 13.0827, 80.2707

# -----------------------------
# NASA API (UNCHANGED)
# -----------------------------
params = {
    "latitude": lat,
    "longitude": lon,
    "start": "20190101",
    "end": "20251231",
    "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS2M",
    "community": "RE",
    "format": "JSON"
}

data = requests.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params).json()
data = data["properties"]["parameter"]

irr = [v for v in data["ALLSKY_SFC_SW_DWN"].values() if v > 0]
temp = [v for v in data["T2M"].values() if v > -50]
humidity = [v for v in data["RH2M"].values() if v >= 0]
wind = [v for v in data["WS2M"].values() if v >= 0]

irradiance_daily = sum(irr)/len(irr)
avg_temp = sum(temp)/len(temp)
avg_humidity = sum(humidity)/len(humidity)
avg_wind = sum(wind)/len(wind)

# -----------------------------
# EFFICIENCY (UNCHANGED)
# -----------------------------
panel_eff = 0.15

temp_loss = 1 + (-0.004*(avg_temp - 25))
wind_gain = 1 + (avg_wind*0.01)
humidity_loss = 1 - (avg_humidity*0.001)

adjusted_eff = panel_eff * temp_loss * wind_gain * humidity_loss
adjusted_eff = max(0.10, min(adjusted_eff, 0.18))

irradiance_yearly = irradiance_daily * 365
eff_factor = adjusted_eff / 0.15

# -----------------------------
# ANALYSIS (UNCHANGED)
# -----------------------------
total_energy = 0
total_panels = 0
total_area = 0
total_system_kw = 0

overlay = image_rgb.copy()

for i, poly in enumerate(polygons):

    roi_mask = np.zeros(mask.shape)
    cv2.fillPoly(roi_mask, [np.array(poly)], 1)

    weighted_pixels = np.sum(mask * roi_mask)
    area = weighted_pixels * (pixel_resolution**2)
    total_area += area

    confidence = np.mean(mask[roi_mask == 1])

    usable_area = area * coverage_factor
    system_kw = min(usable_area / 9, 7)

    confidence_factor = max(0.6, min(confidence, 1.0))
    system_kw *= confidence_factor

    panels = int((system_kw * 1000) / panel_power)
    system_kw = (panels * panel_power) / 1000

    total_system_kw += system_kw

    energy = system_kw * irradiance_yearly * system_loss * eff_factor

    total_energy += energy
    total_panels += panels

    print(f"\nHouse {i+1}")
    print("Area:", round(area,2))
    print("Confidence:", round(confidence,2))
    print("System Size (kW):", round(system_kw,2))
    print("Panels:", panels)
    print("Energy:", int(energy))

    roof_pixels = (mask * roi_mask) > 0.4
    overlay[roof_pixels] = [255, 0, 0]

    M = cv2.moments(np.array(poly))
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) - display_crop
        cy = int(M["m01"] / M["m00"]) - display_crop

        cv2.putText(
            overlay,
            f"H{i+1}",
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

# -----------------------------
# VISUALIZATION (UNCHANGED)
# -----------------------------
img_display = image_rgb[display_crop:-display_crop, display_crop:-display_crop]
mask_display = mask_binary[display_crop:-display_crop, display_crop:-display_crop]
overlay_display = overlay[display_crop:-display_crop, display_crop:-display_crop]

fig, axs = plt.subplots(1, 4, figsize=(16,4))

axs[0].imshow(img_display); axs[0].set_title("Original"); axs[0].axis("off")
axs[1].imshow(mask_display, cmap='gray'); axs[1].set_title("Mask"); axs[1].axis("off")
axs[2].imshow(overlay_display); axs[2].set_title("Selected Houses"); axs[2].axis("off")

axs[3].axis("off")

text = (
    f"Total Area: {total_area:.1f} m²\n"
    f"Total Panels: {total_panels}\n"
    f"Total System Size: {total_system_kw:.2f} kW\n"
    f"Equivalent 1kW Systems: {int(total_system_kw)}\n"
    f"Total Energy: {int(total_energy):,} kWh/year\n\n"
    f"Efficiency: {adjusted_eff:.3f}\n"
    f"Irradiance: {irradiance_daily:.2f}\n"
    f"Temp: {avg_temp:.1f}°C\n"
    f"Wind: {avg_wind:.1f} m/s\n"
    f"Humidity: {avg_humidity:.1f}%"
)

axs[3].text(0, 0.5, text, fontsize=10, va='center')

plt.tight_layout()
plt.show()