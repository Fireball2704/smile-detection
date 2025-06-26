import os
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160
model = tf.keras.models.load_model("saved_model/smile_model.keras")

def predict_folder(folder_path):
    scores = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(folder_path, fname)
        img = Image.open(path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)[0][0]
        scores.append(pred)
        print(f"{fname}: {round(pred, 3)}")
    return scores

print("\n--- Smiling Images ---")
smiling_scores = predict_folder("manual_test/smiling")

print("\n--- Not Smiling Images ---")
not_smiling_scores = predict_folder("manual_test/not_smiling")

print("\n--- Summary ---")
print(f"Smiling avg:     {round(np.mean(smiling_scores), 3)}")
print(f"Not Smiling avg: {round(np.mean(not_smiling_scores), 3)}")
