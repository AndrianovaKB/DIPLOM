from flask import Flask, render_template, request, jsonify
import cv2
import base64
from io import BytesIO
import numpy as np
from model_and_preprocess import DCNN, preprocess_image

app = Flask(__name__)

model = DCNN()
# Загрузка вашей модели
model.load_weights("model_DCNN.weights.h5")

def analyze_emotion(face):
    # Предобработка изображения, если это необходимо
    processed_face = preprocess_image(face)
    # Подача изображения на вход вашей модели
    emotion_probabilities = model.predict(processed_face)
    # Получение эмоции с наибольшей вероятностью
    emotion_index = np.argmax(emotion_probabilities)
    emotions = ["angry", "disgut", "fear", "happy", "neutral", "sad", "surprise"]
    dominant_emotion = emotions[emotion_index]
    # Преобразование вероятностей к типу float
    emotion_probabilities = emotion_probabilities.flatten().astype(float)
    # Создание словаря с вероятностями каждой эмоции в порядке убывания
    emotion_probabilities_sorted = sorted(zip(emotions, emotion_probabilities), key=lambda x: x[1], reverse=True)
    all_emotions = {emotion: round(prob, 4) for emotion, prob in emotion_probabilities_sorted}
    return dominant_emotion, all_emotions

# Глобальные переменные для каскадов Хаара
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Функция для обнаружения лица на изображении
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Если лицо обнаружено, возвращаем координаты его рамки
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y+h, x:x+w]
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.form['image']
    # декодирование изображения из base64 формата
    image_data = base64.b64decode(image_data.split(',')[1])

    # преобразование изображения в формат OpenCV
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Обнаружение лица на изображении
    detected_face_img = detect_face(img)

    # Если лицо не обнаружено, возвращаем ошибку
    if detected_face_img is None:
        return jsonify({'error': 'No face detected'})

    # Ваш код для обработки и рисования квадрата вокруг лица
    # Пример:
    # square_img = draw_square(detected_face_img)

    # Преобразование изображения в base64 для передачи на сервер
    _, buffer = cv2.imencode('.jpg', detected_face_img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    # Вызов функции для определения эмоции
    dominant_emotion, all_emotions = analyze_emotion(detected_face_img)
    print(all_emotions)
    return jsonify({'image': img_str, 'emotion': dominant_emotion, 'emotionProbabilities': all_emotions})


if __name__ == '__main__':
    app.run(debug=True)
