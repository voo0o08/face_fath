import cv2
import numpy as np
import mediapipe as mp
import turtle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance

import matplotlib.pyplot as plt

# ====================
# 1. 이미지 로드
# ====================
image_path = "./data/test3.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# ====================
# 2. Mediapipe 얼굴 인식
# ====================
model_path = r'C:\Users\pdbst\Desktop\face_drawing\mediapipe_study\face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE)
detector = vision.FaceLandmarker.create_from_options(options)
mp_image = mp.Image.create_from_file(image_path)
detection_result = detector.detect(mp_image)

if not detection_result.face_landmarks:
    raise RuntimeError("얼굴이 감지되지 않았습니다.")

landmarks = detection_result.face_landmarks[0]
x151 = int(landmarks[151].x * w)
y151 = int(landmarks[151].y * h)
y_min = y151
x_coords = [int(l.x * w) for l in landmarks]
y_coords = [int(l.y * h) for l in landmarks]
x_min, x_max = min(x_coords), max(x_coords)
y_max = max(y_coords)

# ====================
# 3. skeletonize 함수
# ====================
def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img[:] = eroded[:]
        if cv2.countNonZero(img) == 0:
            break
    return skel

# ====================
# 4. 얼굴 정교화
# ====================
face_crop = image[y_min:y_max, x_min:x_max]
gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
face_edges = cv2.Canny(gray_face, 30, 100)
face_skel = skeletonize(face_edges)

# ====================
# 5. 배경 단순화
# ====================
blurred_image = image.copy()
blurred_image[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(
    blurred_image[y_min:y_max, x_min:x_max], (31, 31), 0)
gray_bg = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
bg_edges = cv2.Canny(gray_bg, 100, 200)
bg_skel = skeletonize(bg_edges)

# ====================
# 6. 합치기 + 마스크 분리
# ====================
final_edges = bg_skel.copy()
final_edges[y_min:y_max, x_min:x_max] = face_skel
mask = np.zeros(final_edges.shape, dtype=np.uint8)
mask[y_min:y_max, x_min:x_max] = 1

# 영역 분리
points_face = np.column_stack(np.where((final_edges > 0) & (mask == 1)))
points_bg = np.column_stack(np.where((final_edges > 0) & (mask == 0)))

# ====================
# 7. 경로 생성 함수
# ====================
def make_path(points, min_dist=1.0):
    if len(points) == 0:
        return []
    visited = np.zeros(len(points), dtype=bool)
    path = []
    current_idx = np.argmin(points[:, 0] + points[:, 1])
    current_point = points[current_idx]
    visited[current_idx] = True
    path.append(tuple(current_point))
    while not np.all(visited):
        dists = distance.cdist([current_point], points[~visited])
        nearest_idx = np.argmin(dists)
        global_idx = np.where(~visited)[0][nearest_idx]
        if dists[0, nearest_idx] < min_dist:
            visited[global_idx] = True
            continue
        current_point = points[global_idx]
        visited[global_idx] = True
        path.append(tuple(current_point))
    return path

# 경로 생성
path_face = make_path(points_face, min_dist=1.0)
path_bg = make_path(points_bg, min_dist=1.0)

# 샘플링 비율 적용
def sample_path(path, every_n):
    return path[::every_n]

path_face_sampled = sample_path(path_face, 1)  # 얼굴은 2점마다
path_bg_sampled = sample_path(path_bg, 8)      # 배경은 3점마다
full_path = path_face_sampled + path_bg_sampled

# --------------------
# 포인트 수 출력
# --------------------
print(f"얼굴 포인트 개수 (원본): {len(path_face)}")
print(f"배경 포인트 개수 (원본): {len(path_bg)}")
print(f"얼굴 샘플링 후 개수 (2점마다): {len(path_face_sampled)}")
print(f"배경 샘플링 후 개수 (3점마다): {len(path_bg_sampled)}")
print(f"전체 그릴 포인트 수: {len(path_face_sampled) + len(path_bg_sampled)}")

# --------------------
# 시각화
# --------------------
vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

for y, x in path_face_sampled:
    cv2.circle(vis_image, (x, y), 1, (255, 0, 0), -1)  # 빨간색: 얼굴
for y, x in path_bg_sampled:
    cv2.circle(vis_image, (x, y), 1, (0, 255, 0), -1)  # 초록색: 배경

plt.figure(figsize=(10, 8))
plt.imshow(vis_image)
plt.title("Sampled Drawing Points (Red=Face, Green=Background)")
plt.axis("off")
plt.show()



# ====================
# 8. Turtle로 그리기
# ====================
screen = turtle.Screen()
screen.setup(width=800, height=600)
turtle.speed(0)
turtle.bgcolor("white") 
scale_x = 800 / w
scale_y = 600 / h
BREAK_DIST = 20.0

turtle.penup()
for i, (y, x) in enumerate(full_path):
    px, py = x * scale_x - 400, 300 - y * scale_y
    if i == 0:
        turtle.goto(px, py)
        turtle.pendown()
    else:
        prev_y, prev_x = full_path[i-1]
        dist = np.linalg.norm(np.array((x, y)) - np.array((prev_x, prev_y)))
        if dist > BREAK_DIST:
            turtle.penup()
        turtle.goto(px, py)
        if dist > BREAK_DIST:
            turtle.pendown()

turtle.done()
