import cv2
import numpy as np
import tensorflow as tf
import time

# --------------------------------------------------
# 0.  CONFIG
# --------------------------------------------------
MODEL_PATH = "saved_model/smile_model.keras"
IMG_SIZE   = 160
THRESHOLD  = 0.55       # tweak if needed
MIN_FACE   = 80 * 80    # ignore faces < 80×80 px  (OPTIONAL)

# --------------------------------------------------
# 1.  LOAD MODEL & CASCADE
# --------------------------------------------------
model        = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# 2.  START CAMERA
# --------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

print("✅ Smile detector running — press Q to quit")
smile_count = 0               # (OPTIONAL) running total
prev_state  = False           #  was smiling last frame?

# --------------------------------------------------
# 3.  MAIN LOOP
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed. Exiting …")
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w * h < MIN_FACE:           #  too small ⇒ skip (OPTIONAL)
            continue

        roi = frame[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = np.expand_dims(roi, 0)

        score = model.predict(roi, verbose=0)[0][0]
        is_smile = score > THRESHOLD

        # (OPTIONAL) count smiles when switching from not-smile → smile
        if is_smile and not prev_state:
            smile_count += 1
        prev_state = is_smile

        label = f"{'Smiling 😀' if is_smile else 'Not Smiling 😐'}  ({score:.2f})"
        color = (0, 255, 0) if is_smile else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # (OPTIONAL) display the running smile count in top-left
    cv2.putText(frame, f"Total smiles: {smile_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Smile Detector — press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------------------------
# 4.  CLEAN-UP
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print(f"👋 Session ended. Total smiles detected: {smile_count}")
