'''
drawing2_txt -> drawing2를 텍스트 파일로 저장하는 파일
'''
import cv2
import numpy as np
from scipy.spatial import distance

# 파라미터
MIN_DIST = 3.0  # 최소 점 간 거리
IMG_PATH = "./data/test.jpg"

# 좌표 변환 범위
X_MIN, X_MAX = -800, -650
Y_MIN, Y_MAX = -400, -200

# 이미지 불러오기 및 전처리
image = cv2.imread(IMG_PATH)
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Skeletonization 함수
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

# (y, x) 좌표 추출 및 경로 최적화
points = np.column_stack(np.where(skeleton > 0))
visited = np.zeros(len(points), dtype=bool)
path = []

current_idx = np.argmin(points[:, 0] + points[:, 1])
current_point = points[current_idx]
visited[current_idx] = True
path.append(tuple(current_point))

while not np.all(visited):
    dists = distance.cdist([current_point], points[~visited])
    nearest_idx_in_unvisited = np.argmin(dists)
    global_idx = np.where(~visited)[0][nearest_idx_in_unvisited]
    if dists[0, nearest_idx_in_unvisited] < MIN_DIST:
        visited[global_idx] = True
        continue
    current_point = points[global_idx]
    visited[global_idx] = True
    path.append(tuple(current_point))

# draw_img에 그려진 점만 수집
drawn_points_set = set(tuple(pt) for pt in path)

# path 분할 (선별 경계 기준)
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

lines = split_path_into_lines(path)

# 텍스트 저장용 좌표 변환 함수
h, w = gray.shape
def convert_coord(x, y):
    new_x = ((x / w) * (X_MAX - X_MIN)) + X_MIN
    new_y = ((y / h) * (Y_MAX - Y_MIN)) + Y_MIN
    return round(new_x, 2), round(new_y, 2)

# 텍스트 라인 생성
output_lines = []
for line in lines:
    if len(line) < 2:
        continue
    start_y, start_x = line[0]
    sx, sy = convert_coord(start_x, start_y)
    output_lines.append(f"{sx},{sy},170.0,90.03,0.35,-86.91")
    for y, x in line[1:-1]:
        if (y, x) in drawn_points_set:
            cx, cy = convert_coord(x, y)
            output_lines.append(f"{cx},{cy},160.0,90.03,0.35,-86.91")
    end_y, end_x = line[-1]
    ex, ey = convert_coord(end_x, end_y)
    output_lines.append(f"{ex},{ey},170.0,90.03,0.35,-86.91")

# 파일로 저장
output_path = "./data/turtle_coordinates.txt"
with open(output_path, "w") as f:
    for line in output_lines:
        f.write(line + "\n")

output_path