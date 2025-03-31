'''
저장된 txt 파일을 읽어 turtle로 그림을 그리는 파일
'''
import turtle

# 원래 이미지 크기
IMG_WIDTH, IMG_HEIGHT = 800, 600

# 정규화 좌표 범위
X_MIN, X_MAX = -800, -650
Y_MIN, Y_MAX = -400, -200

# TXT 파일 경로
TXT_PATH = "./data/turtle_coordinates.txt"

# Turtle 설정
screen = turtle.Screen()
screen.setup(width=800, height=600)
screen.bgcolor("white")
turtle.speed(0)
turtle.pensize(1)
colors = ["black", "red", "blue", "green", "purple", "orange"]
color_idx = 0

# 좌표 역변환 함수
def reverse_transform(x, y):
    img_x = int(((x - X_MIN) / (X_MAX - X_MIN)) * IMG_WIDTH)
    img_y = int(((y - Y_MIN) / (Y_MAX - Y_MIN)) * IMG_HEIGHT)
    screen_x = img_x - 400  # turtle 좌표계 변환
    screen_y = 300 - img_y
    return screen_x, screen_y

# 파일에서 좌표 읽기
with open(TXT_PATH, "r") as f:
    lines = f.readlines()

pen_down = False
for line in lines:
    parts = line.strip().split(",")
    if len(parts) < 6:
        continue

    x, y, flag, *_ = map(float, parts)
    screen_x, screen_y = reverse_transform(x, y)

    if flag == 170.0:
        turtle.penup()
        turtle.goto(screen_x, screen_y)
        turtle.pencolor(colors[color_idx % len(colors)])
        turtle.pendown()
        color_idx += 1
    else:
        turtle.goto(screen_x, screen_y)

turtle.done()
