import cv2

# Загрузка предварительно обученного классификатора для детекции лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

while True:
    # Считывание кадра
    ret, frame = cap.read()
    
    # Преобразование кадра в оттенки серого (для повышения производительности)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Отрисовка прямоугольника вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Отображение кадра
    cv2.imshow('Face Detection', frame)
    
    # Обновление кадра (если не сделать это, окно может зависнуть)
    cv2.waitKey(1)
    
    # Ожидание нажатия клавиши 'q' для выхода из программы
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
