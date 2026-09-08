"""Measure text band positions in source ARMED image at native resolution."""
import cv2
import numpy as np

SRC = "E:/迁移/Documents/My Pictures/Saved Pictures/歪歪.library/images/MTI8BUIRMMKHF.info/b78e60de8dfdf44acda99395326a7298.jpg"
img = cv2.imread(SRC)
H, W = img.shape[:2]
print(f"Native size: {W}x{H}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Per-row dark pixel count (text bands = rows with many dark pixels)
dark_per_row = (gray < 75).sum(axis=1)
# Find continuous regions where dark count > 100 (text bands)
in_band = dark_per_row > 80
bands = []
start = None
for y in range(H):
    if in_band[y] and start is None:
        start = y
    elif not in_band[y] and start is not None:
        bands.append((start, y, y - start, dark_per_row[start:y].max()))
        start = None
if start is not None:
    bands.append((start, H, H - start, dark_per_row[start:H].max()))

print("All text-like bands (start_y, end_y, height, peak_dark_count):")
for b in bands:
    print(f"  y={b[0]:4d}..{b[1]:4d}  h={b[2]:3d}  peak={b[3]}")

# Look at the three main text bands: WE SUPPORT THE, ARMED, FORCES
# The dog tag starts around y=700, so text bands are above that
print("\nFiltered for text bands only (above dog tag y=700, height>20):")
for b in bands:
    if b[1] < 700 and b[2] > 20:
        print(f"  y={b[0]:4d}..{b[1]:4d}  h={b[2]:3d}  relative=({b[0]/H:.3f}, {b[1]/H:.3f})")

# Now measure ARMED letter height: find the "ARMED" band, then find its actual letter height within
print("\nMeasuring 'ARMED' band letter geometry:")
# ARMED is the largest text. From the bands above, pick the band with peak count
