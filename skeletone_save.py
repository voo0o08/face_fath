import cv2
import numpy as np
import turtle

# 이미지 로드
image_path = "./data/test2.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다. 다시 업로드해주세요.")

# 블러를 사용해 노이즈 제거
'''
양방향 필터 (Bilateral Filter)는 가우시안 필터를 사용하여 노이즈를 제거하면서도 엣지를 보존하는 필터
'''
bilateral = cv2.bilateralFilter(image, -1, 75, 75)

# 그레이스케일 변환
'''
밑에서 사용하는 canny 인자에 image를 그레이스케일로 사용함 
'''
gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)

# Canny 엣지 검출
'''
edges = cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
'''
edges = cv2.Canny(gray, 50, 150)

# Skeletonization 적용 (한 줄의 선만 남기기)
def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)  # 빈 이미지 생성
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 3x3 구조화 요소 생성

    while True:
        open_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)  # Morphological Opening 적용 / Opening = Erosion(침식) + Dilation(팽창)
        temp = cv2.subtract(image, open_img)  # Edge 부분만 추출
        eroded = cv2.erode(image, element)  # Erosion (침식) 적용
        skel = cv2.bitwise_or(skel, temp)  # 결과 누적

        # Skeletonization 진행 과정 확인
        cv2.imshow("Skeletonization Process", skel)
        key = cv2.waitKey(0)  # 키 입력 대기
        if key == 27:  # ESC 키를 누르면 종료
            break
        

        image[:] = eroded[:]  # 다음 단계로 업데이트

        if cv2.countNonZero(image) == 0:  # 이미지가 완전히 사라지면 종료
            break

    cv2.destroyAllWindows()
    return skel

edges_skeleton = skeletonize(edges)
cv2.imwrite("Skeletonization.png", edges_skeleton)

# # Contour 검출 (중복된 선 제거)
# contours, _ = cv2.findContours(edges_skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Contour 단순화하여 중복 제거
# simplified_contours = []
# for contour in contours:
#     epsilon = 0.005 * cv2.arcLength(contour, True)  # 단순화 정도 조절
#     simplified = cv2.approxPolyDP(contour, epsilon, True)
#     if len(simplified) > 5:  # 너무 작은 선은 제거
#         simplified_contours.append(simplified)

# # 흰 배경 이미지 생성
# contour_img = np.ones_like(gray) * 255  # 단일 채널로 만듦 (흰 배경)

# # 윤곽선 그리기
# cv2.drawContours(contour_img, simplified_contours, -1, (0), 1)  # 검은색(0), 두께 1

# # 컬러로 변환 (imshow와 저장용)
# contour_img_color = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)

# # 이미지 출력
# cv2.imshow("Outlined Image", contour_img_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 이미지 저장
# # cv2.imwrite("contour_output.png", contour_img_color)