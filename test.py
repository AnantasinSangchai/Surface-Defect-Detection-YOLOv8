from ultralytics import YOLO

model = YOLO(r"C:\Project\YOLO\runs\detect\train5\weights\best.pt")

results = model.predict(source=r"C:\Project\YOLO\datasets\val\images\scratches_70.jpg", save=True, conf=0.25)

results[0].show()