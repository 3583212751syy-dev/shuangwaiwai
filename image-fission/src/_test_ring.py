import numpy as np

def _ring_sector_mask(h, w, c_x, c_y, r_in, r_out, a0, a1):
    ys, xs = np.ogrid[:h, :w]
    d2 = (xs - c_x) ** 2 + (ys - c_y) ** 2
    in_ring = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    sweep = (a1 - a0) % 360
    if sweep == 0 and (a1 - a0) != 0:
        sweep = 360
    if sweep >= 359.99:
        return in_ring
    ang = (np.degrees(np.arctan2(ys - c_y, xs - c_x)) - a0) % 360
    in_angle = ang <= sweep
    return in_ring & in_angle

h, w = 2000, 1552
mask = _ring_sector_mask(h, w, 776, 746, 0, 326, 0, 360)
print('sum', mask.sum(), 'expected ~333000')
