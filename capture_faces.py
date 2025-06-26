import cv2
import os

# Set folder and label
label = input("Are you smiling? (yes/no): ").strip().lower()
folder = "manual_test/smiling" if label == "yes" else "manual_test/not_smiling"

# Ensure folder exists
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save image, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display window
    cv2.imshow("Capture Your Face", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        path = os.path.join(folder, f"{label}_{count}.jpg")
        cv2.imwrite(path, frame)
        print(f"Saved: {path}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
