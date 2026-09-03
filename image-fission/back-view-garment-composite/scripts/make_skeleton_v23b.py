"""
v23-b: 从 front_model.jpg 提取 OpenPose 骨架（mediapipe）→ 水平翻转
      **不画任何脸部关键点**（鼻子/眼睛/耳朵都不画），强制纯背
"""
import cv2
import numpy as np
import mediapipe as mp

front = cv2.imread('ComfyUI/input/front_model.jpg')
Hf, Wf = front.shape[:2]
print(f'front {Wf}x{Hf}')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.5)
rgb = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
res = pose.process(rgb)
assert res.pose_landmarks, 'no pose detected'
lm = res.pose_landmarks.landmark

TARGET_W, TARGET_H = 1024, 1536
canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)

# ONLY body+limbs, NO face keypoints
KEY = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_index': 19, 'right_index': 20,
    'left_pinky': 17, 'right_pinky': 18,
}

def to_canvas(idx):
    p = lm[idx]
    x = int((1.0 - p.x) * TARGET_W)  # horizontal mirror for back view
    y = int(p.y * TARGET_H)
    return x, y

def get(name):
    idx = KEY[name]
    p = lm[idx]
    return to_canvas(idx), p.visibility > 0.5

BONES = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]

for a, b in BONES:
    pa, va = get(a); pb, vb = get(b)
    if va and vb:
        cv2.line(canvas, pa, pb, (255, 255, 255), 4)
for name in KEY:
    pa, va = get(name)
    if va:
        cv2.circle(canvas, pa, 6, (255, 255, 255), -1)

cv2.imwrite('ComfyUI/input/back_pose_skeleton_v23b.png', canvas)
print('saved back_pose_skeleton_v23b.png (no face, mirrored, 1024x1536)')
for name in ('left_shoulder', 'right_shoulder', 'left_hip', 'left_ankle', 'right_ankle'):
    pa, va = get(name)
    print(f'  {name}: {pa} vis={va}')
pose.close()
