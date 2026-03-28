from ultralytics import YOLO
import cv2

# Load model (make sure best.pt is in same folder)
model = YOLO("yolov8n.pt")

# Load image
image_path = "test.jpg"
img = cv2.imread(image_path)

# Run detection
results = model(img)

# Show results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
