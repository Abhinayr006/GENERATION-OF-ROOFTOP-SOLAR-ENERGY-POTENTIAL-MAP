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