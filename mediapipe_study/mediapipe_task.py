import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 모델 경로 설정
model_path  = r'C:\Users\pdbst\Desktop\face_drawing\mediapipe_study\face_landmarker.task'

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE) # 필수인자 : base_options, running_mode 
detector = vision.FaceLandmarker.create_from_options(options) # 디텍터 생성 

# STEP 3: Load the input image.
# 이미지는 numpy array로 변환하여 사용 or mediapipe의 Image 객체로 변환하여 사용
# Load the input image from an image file.
image_path = "./data/test3.jpg"
image = mp.Image.create_from_file(image_path)

# STEP 4: 랜드마크 검출~
detection_result = detector.detect(image)

# STEP 5: 결과 시각화 (Matplotlib + OpenCV)
image_np = cv2.imread(image_path)
h, w, _ = image_np.shape

if not detection_result.face_landmarks: 
    print("얼굴이 감지되지 않았습니다.")
else:
    for face_landmarks in detection_result.face_landmarks:
        for idx, landmark in enumerate(face_landmarks):
            x_px = int(landmark.x * w)
            y_px = int(landmark.y * h)
            cv2.circle(image_np, (x_px, y_px), 1, (0, 255, 0), -1)
            cv2.putText(image_np, str(idx), (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.2, (255, 0, 0), 1, cv2.LINE_AA)
            print(idx)

    # 시각화
plt.figure(figsize=(12, 10))  # <- 크기 키움
plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.title("Face Landmarks with Index")
plt.axis("off")
plt.show()