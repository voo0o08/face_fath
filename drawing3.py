import cv2
import numpy as np
import mediapipe as mp
import turtle
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================
# 1. 이미지 로드
# ====================
image_path = "./data/test2.png"
image_path = "./data/test3.jpg"

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# ====================
# 2. Mediapipe Mesh 기반 얼굴 인식
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

# 151번 좌표를 기준으로 y_min 결정
landmarks = detection_result.face_landmarks[0]
x151 = int(landmarks[151].x * w)
y151 = int(landmarks[151].y * h)
y_min = y151

# Bounding box 생성
x_coords = [int(l.x * w) for l in landmarks]
y_coords = [int(l.y * h) for l in landmarks]
x_min, x_max = min(x_coords), max(x_coords)
y_max = max(y_coords)

# ====================
# 3. 얼굴 영역 정교화
# ====================
face_crop = image[y_min:y_max, x_min:x_max]
gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
face_edges = cv2.Canny(gray_face, 30, 100)

# skeletonize 함수 정의
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

face_skel = skeletonize(face_edges)

# ====================
# 4. 배경 영역 단순화 (blur 처리)
# ====================
blurred_image = image.copy()
blurred_image[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(
    blurred_image[y_min:y_max, x_min:x_max], (31, 31), 0)
gray_bg = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
bg_edges = cv2.Canny(gray_bg, 100, 200)

# ====================
# 5. 합치기
# ====================
final_edges = bg_edges.copy()
final_edges[y_min:y_max, x_min:x_max] = face_skel

# ====================
# 6. Contour로 path 만들기
# ====================
contours, _ = cv2.findContours(final_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ====================
# 7. Turtle로 그리기
# ====================
screen = turtle.Screen()
screen.setup(width=800, height=600)
turtle.speed(0)
turtle.bgcolor("white")
turtle.pensize(1)
scale_x = 800 / w
scale_y = 600 / h

for contour in contours:
    if len(contour) < 5:
        continue
    turtle.penup()
    x, y = contour[0][0]
    turtle.goto(x * scale_x - 400, 300 - y * scale_y)
    turtle.pendown()
    for point in contour:
        x, y = point[0]
        turtle.goto(x * scale_x - 400, 300 - y * scale_y)

turtle.done()
