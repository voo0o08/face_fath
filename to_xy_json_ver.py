import cv2
import numpy as np
import json

# 이미지 경로
image_path = "data/test2_result1.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다. 다시 업로드해주세요.")

# 블러를 사용해 노이즈 제거
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

# 그레이스케일 변환
gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

# Canny 엣지 검출
edges = cv2.Canny(gray, 50, 150)

# Skeletonization 함수
def skeletonize(image):
    skel = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(image, open_img)
        eroded = cv2.erode(image, element)
        skel = cv2.bitwise_or(skel, temp)
        image[:] = eroded[:]
        if cv2.countNonZero(image) == 0:
            break
    return skel

# Skeletonization 적용
edges_skeleton = skeletonize(edges)

# 윤곽선 검출 및 단순화
contours, _ = cv2.findContours(edges_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
simplified_contours = []
for contour in contours:
    epsilon = 0.005 * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    if len(simplified) > 5:
        simplified_contours.append(simplified)

# 전체 좌표 범위 설정
X_MIN, X_MAX = -800, -650
Y_MIN, Y_MAX = -400, -200

# 모든 contour 좌표 수집 후 min/max 계산
all_points = np.vstack([cnt.reshape(-1, 2) for cnt in simplified_contours])
x_min_img, y_min_img = np.min(all_points, axis=0)
x_max_img, y_max_img = np.max(all_points, axis=0)

# JSON 데이터 구성
result = {}
for idx, contour in enumerate(simplified_contours):
    points = []
    for idx2, point in enumerate(contour):
        
        x, y = point[0]

        # 선형 보간으로 좌표 변환
        new_x = ((x - x_min_img) / (x_max_img - x_min_img)) * (X_MAX - X_MIN) + X_MIN
        new_y = ((y - y_min_img) / (y_max_img - y_min_img)) * (Y_MAX - Y_MIN) + Y_MIN
        if idx2 == 0:
            points.append((round(new_x, 2), round(new_y, 2), 170.0, 90.03, 0.35, -86.91)) # 시작 
            
        points.append((round(new_x, 2), round(new_y, 2), 160.0, 90.03, 0.35, -86.91))
        
    points.append((round(new_x, 2), round(new_y, 2), 170.0, 90.03, 0.35, -86.91)) # 끝 

    result[f"line{idx}"] = {
        "points": points,
        "color_idx": idx
    }

# JSON 저장
output_path = "./data/converted_lines.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

output_path
