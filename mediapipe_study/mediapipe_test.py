import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "./data/test2.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# 얼굴 랜드마크 탐지
with mp_face_mesh.FaceMesh(static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(image_rgb)

# 얼굴 랜드마크 그리기
annotated_image = image_rgb.copy()
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
        )
else:
    raise RuntimeError("얼굴이 인식되지 않았습니다.")

# 결과 시각화
plt.figure(figsize=(8, 8))
plt.imshow(annotated_image)
plt.title("Mediapipe Face Mesh")
plt.axis("off")
plt.show()
