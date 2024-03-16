import os
import dlib
import cv2
import json

# Создаем детектор лиц
face_detector = dlib.get_frontal_face_detector()
# Создаем предиктор для поиска ключевых точек
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Папка с изображениями
images_folder = "reference_images"

# Создаем папку для сохранения ключевых точек
keypoints_folder = "keypoints"
if not os.path.exists(keypoints_folder):
    os.makedirs(keypoints_folder)

# Проходим по всем файлам в папке с изображениями
for i, filename in enumerate(os.listdir(images_folder), start=1):
    image_path = os.path.join(images_folder, filename)

    # Загружаем изображение
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Обнаруживаем лица на изображении
    faces = face_detector(gray)

    # Проходим по всем обнаруженным лицам
    for face in faces:
        # Находим ключевые точки для лица
        landmarks = shape_predictor(gray, face)

        # Создаем словарь для хранения координат ключевых точек
        keypoints_dict = {}
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            keypoints_dict[f"point_{n+1}"] = (x, y)

        # Сохраняем координаты ключевых точек в файл JSON
        keypoints_file_path = os.path.join(keypoints_folder, f"key_cord_{i}.json")
        with open(keypoints_file_path, "w") as f:
            json.dump(keypoints_dict, f)

        print(f"Ключевые точки были созданы для {filename}.")

print("Все ключевые точки были успешно созданы.")