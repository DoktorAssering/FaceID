import cv2
import dlib
import os
import time

# Загрузка предварительно обученного классификатора для детекции лиц
face_detector = dlib.get_frontal_face_detector()
# Загрузка предварительно обученной модели для поиска ключевых точек лица
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Загрузка видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

# Папка с файлами ключевых точек
keypoints_folder = "keypoints"

# Получаем список файлов в папке keypoints
keypoints_files = os.listdir(keypoints_folder)

while True:
    # Считывание кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого (для повышения производительности)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_detector(gray)
    
    # Проходим по всем обнаруженным лицам
    for face in faces:
        print("Начало считывания лица...")
        time.sleep(1)
        # Поиск ключевых точек для каждого обнаруженного лица
        landmarks = shape_predictor(gray, face)
        
        # Проходим по всем файлам с ключевыми точками
        for file in keypoints_files:
            # Считываем координаты ключевых точек из файла
            with open(os.path.join(keypoints_folder, file), 'r') as f:
                keypoints_data = f.readlines()

            # Преобразуем координаты из строк в список кортежей целых чисел
            keypoints = [tuple(map(int, point.split())) for point in keypoints_data]
            
            # Проверяем, совпадают ли ключевые точки с найденными лицами
            if all([(landmarks.part(n).x, landmarks.part(n).y) in keypoints for n in range(68)]):
                indicator_color = (0, 255, 0)
                print("Считывание прошло успешно.")
                # Здесь выполняем дальнейший код
            else:
                indicator_color = (0, 0, 255)
                print("Лицо не схоже с владельцем этого устройства.")
        
        # Рисуем прямоугольник вокруг обнаруженного лица
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), indicator_color, 2)
        
    # Отображение кадра
    cv2.imshow('Facial Landmarks Detection', frame)
    
    # Клавиша для выхода из программы
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()