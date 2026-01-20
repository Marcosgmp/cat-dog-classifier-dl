import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/cat_dog_model.h5")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam")
    exit()

print("Pressione 'q' para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img, verbose=0)[0][0]

    if prediction > 0.6:
        label = f"Cachorro ({prediction*100:.2f}%)"
        color = (0, 255, 0)
    elif prediction <= 0.4:
        label = f"Gato ({(1-prediction)*100:.2f}%)"
        color = (255, 0, 0)
    else:
        confidence = max(prediction, 1 - prediction) * 100
        label = f"Nao identificado ({confidence:.2f}%)"
        color = (0, 255, 255)


    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Cat vs Dog - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()