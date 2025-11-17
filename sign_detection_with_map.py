from ultralytics import YOLO
import geocoder
import folium
import json
import os

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Memory file
MEM_FILE = "sign_memory.json"

# Load existing memory or create new
if os.path.exists(MEM_FILE):
    with open(MEM_FILE, "r") as f:
        memory_data = json.load(f)
else:
    memory_data = []

def get_gps():
    loc = geocoder.ip("me")
    return loc.latlng  # returns (lat, lon)

def save_memory(entry):
    memory_data.append(entry)
    with open(MEM_FILE, "w") as f:
        json.dump(memory_data, f, indent=4)

def update_map(lat, lon, label):
    # Center map on Michigan
    michigan_center = [44.3148, -85.6024]  # Michigan center point

    mymap = folium.Map(location=[lat, lon], zoom_start=14)

    # Add detected sign marker
    folium.Marker(
        [lat, lon],
        popup=f"Detected: {label}",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(mymap)

    # Save map
    mymap.save("sign_map.html")
    print("Map updated → sign_map.html")

def detect_sign_and_map(image_path):
    print("🔍 Running YOLO detection...")
    results = model(image_path)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = float(box.conf[0])

        print(f"✔ Detected sign: {label} ({conf:.2f})")

        # Get GPS
        lat, lon = get_gps()
        print(f"📍 GPS Location: {lat}, {lon}")

        # Save in memory
        entry = {
            "sign_type": label,
            "confidence": conf,
            "latitude": lat,
            "longitude": lon
        }
        save_memory(entry)

        # Update map
        update_map(lat, lon, label)

if __name__ == "__main__":
    detect_sign_and_map(r"D:\WSU Academy Files\Fall 2025\ECE 5995\Final_Project\50mph.jpg")
