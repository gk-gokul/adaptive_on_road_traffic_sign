from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    yaml_path = r"D:\WSU Academy Files\Fall 2025\ECE 5995\Final_Project\dataset\challenging-dev\challenging\YOLOv8\data.yaml"
    
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0
    )
    print("Done")

if __name__ == "__main__":
    main()
