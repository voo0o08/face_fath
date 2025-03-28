import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# 파라미터
MIN_DIST = 3.0  # 최소 점 간 거리
IMG_PATH = "./data/test.jpg"

# 이미지 불러오기 및 전처리
image = cv2.imread(IMG_PATH)
if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

bilateral = cv2.bilateralFilter(image, 9, 75, 75)
gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Skeletonization
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

skeleton = skeletonize(edges)

# (y, x) 좌표 추출
points = np.column_stack(np.where(skeleton > 0))
visited = np.zeros(len(points), dtype=bool)
path = []

# 시작점: 가장 왼쪽 위 점
current_idx = np.argmin(points[:, 0] + points[:, 1])
current_point = points[current_idx]
visited[current_idx] = True
path.append(tuple(current_point))

# TSP-like 연결 (가장 가까운 점 찾기)
while not np.all(visited):
    dists = distance.cdist([current_point], points[~visited])
    nearest_idx_in_unvisited = np.argmin(dists)
    global_idx = np.where(~visited)[0][nearest_idx_in_unvisited]
    if dists[0, nearest_idx_in_unvisited] < MIN_DIST:
        visited[global_idx] = True
        continue  # 너무 가까우면 스킵
    current_point = points[global_idx]
    visited[global_idx] = True
    path.append(tuple(current_point))

# 결과 이미지 생성
draw_img = np.ones_like(image) * 255
for pt in path:
    cv2.circle(draw_img, (pt[1], pt[0]), 1, (0, 0, 0), -1)

# 저장 및 확인
cv2.imwrite("path_thinned.png", draw_img)
plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
plt.title("Optimized Drawing Path")
plt.axis("off")
plt.show()

#############################
# path를 여러 개의 선으로 나누고, turtle로 뗐다 붙였다 그리는 버전
import turtle

# 거리 기준 (너무 멀면 새 선으로 인식)
BREAK_DIST = 15.0

# path를 여러 선으로 분할
def split_path_into_lines(path, break_dist=15.0):
    lines = []
    current_line = [path[0]]
    for prev, curr in zip(path[:-1], path[1:]):
        dist = np.linalg.norm(np.array(curr) - np.array(prev))
        if dist > break_dist:
            if len(current_line) > 1:
                lines.append(current_line)
            current_line = [curr]
        else:
            current_line.append(curr)
    if len(current_line) > 1:
        lines.append(current_line)
    return lines

# 좌표 변환용
h, w = 600, 800
scale_x = 800 / w
scale_y = 600 / h

# Turtle 설정
turtle.speed(0)
turtle.bgcolor("white")
turtle.pensize(1)
screen = turtle.Screen()
screen.setup(width=800, height=600)
colors = ["black", "red", "blue", "green", "purple", "orange"]

# 선 분할
lines = split_path_into_lines(path)

# 그리기
for idx, line in enumerate(lines):
    if len(line) == 0:
        continue
    turtle.penup()
    y, x = line[0]
    turtle.goto(x * scale_x - 400, 300 - y * scale_y)
    turtle.pencolor(colors[idx % len(colors)])
    turtle.pendown()
    for y, x in line[1:]:
        turtle.goto(x * scale_x - 400, 300 - y * scale_y)

turtle.done()
