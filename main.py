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

# def detect_face(image):
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         face = image[y:y + h, x:x + w]
#         return face, (x, y, w, h)
#     return None, None
#
def analyze_emotion(face):
    # Предобработка изображения, если это необходимо
    processed_face = preprocess_image(face)
    # Подача изображения на вход вашей модели
    emotion_probabilities = model.predict(processed_face)
    # Получение эмоции с наибольшей вероятностью
    emotion_index = np.argmax(emotion_probabilities)
    emotions = ["angry", "disgut", "fear", "happy", "neutral", "sad", "surprise"]
    emotion = emotions[emotion_index]
    return emotion
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     if 'file' not in request.files:
#         return "No file part"
#
#     file = request.files['file']
#
#     image_stream = BytesIO()
#     file.save(image_stream)
#     image_stream.seek(0)
#     file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
#     frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#
#     face, (x, y, w, h) = detect_face(frame)
#
#     if face is not None:
#         emotion = analyze_emotion(file)
#         cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#         _, img_encoded = cv2.imencode('.jpg', frame)
#         img_base64 = base64.b64encode(img_encoded).decode('utf-8')
#         result = {'region': 'data:image/jpeg;base64,' + img_base64, 'dominant_emotion': emotion}
#         return render_template('result.html', result=result)
#     else:
#         return "No face detected"
#
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


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

    print(detected_face_img)
    # Вызов функции для определения эмоции
    emotion = analyze_emotion(detected_face_img)

    return jsonify({'image': img_str, 'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
