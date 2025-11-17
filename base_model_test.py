from ultralytics import YOLO

model = YOLO(r"D:\WSU Academy Files\Fall 2025\ECE 5995\Final_Project\runs\detect\train\weights\best.pt")
results = model(r"50mph.jpg")
