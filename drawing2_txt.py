import cv2
import numpy as np
import mediapipe as mp
import turtle
import matplotlib.pyplot as plt

# ====================
# 1. 이미지 로드
# ====================
image_path = "./data/test.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# ====================
# 2. Mediapipe 얼굴 인식
# ====================
mp_face = mp.solutions.face_mesh
with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(image_rgb)

if not results.multi_face_landmarks:
    raise RuntimeError("얼굴을 찾을 수 없습니다.")

# ====================
# 3. Landmark 기준 얼굴 영역 가져오기
# ====================
landmarks = results.multi_face_landmarks[0].landmark
xs = [int(lm.x * w) for lm in landmarks]
ys = [int(lm.y * h) for lm in landmarks]
x_min = max(0, min(xs) - 10)
x_max = min(w, max(xs) + 10)
y_min = max(0, min(ys) - 10)
y_max = min(h, max(ys) + 10)

# 얼굴 특징점 시각화 (눈, 코, 입)
landmark_img = image.copy()
for x, y in zip(xs, ys):
    cv2.circle(landmark_img, (x, y), 1, (0, 255, 0), -1)

cv2.imshow("Face landmarks", landmark_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ====================
# 4. 얼굴 영역 정교화
# ====================
face_crop = image[y_min:y_max, x_min:x_max]
gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
face_edges = cv2.Canny(gray_face, 30, 100)

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
# 5. 배경 단순화
# ====================
mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
mask[y_min:y_max, x_min:x_max] = 0
background = cv2.bitwise_and(image, image, mask=mask)
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
bg_edges = cv2.Canny(gray_bg, 100, 200)

cv2.imshow("bg_edges", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ====================
# 6. 합치기
# ====================
final_edges = np.zeros_like(gray_bg)
final_edges[y_min:y_max, x_min:x_max] = face_skel
final_edges += bg_edges

# ====================
# 7. Contour 추출 및 Turtle 그리기
# ====================
contours, _ = cv2.findContours(final_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

screen = turtle.Screen()
screen.setup(width=800, height=600)
turtle.speed(0)
turtle.bgcolor("white")
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