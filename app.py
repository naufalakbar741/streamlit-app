import streamlit as st
import cv2
import numpy as np
import time
import requests
from PIL import Image
from io import BytesIO
import threading
from ultralytics import YOLO
import queue
import paho.mqtt.client as mqtt
import json

st.set_page_config(
    page_title="GrowNet Dashboard",
    page_icon="🌱",
    layout="wide"
)

ESP_CAM_URL = "http://192.168.127.192/"

MQTT_SERVER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "streamlit_controller"
MQTT_TOPIC = "sam/esp32/starter"

@st.cache_resource
def load_model():
    return YOLO('best.pt')

mqtt_client = None

def initialize_mqtt_client():
    global mqtt_client
    
    if mqtt_client is not None:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        except:
            pass
    
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv311, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {MQTT_SERVER}:{MQTT_PORT}")
        else:
            print(f"Failed to connect with result code {rc}")
    
    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print(f"Unexpected disconnection with result code {rc}")
    
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
    try:
        client.connect(MQTT_SERVER, 1883)
        client.loop_start()
        mqtt_client = client
        return True
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        return False

def send_mqtt_message():
    global mqtt_client
    
    if mqtt_client is None or not mqtt_client.is_connected():
        if not initialize_mqtt_client():
            st.toast("Failed to connect to MQTT broker", icon="❌")
            return
    
    try:
        message = json.dumps({"msg": 1})
        result = mqtt_client.publish(MQTT_TOPIC, message)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            st.toast("Message sent successfully!", icon="✅")
            print(f"Message published to {MQTT_TOPIC}: {message}")
        else:
            st.toast(f"Failed to send message: {mqtt.error_string(result.rc)}", icon="❌")
            print(f"Failed to publish message: {result}")
            
            initialize_mqtt_client()
            
    except Exception as e:
        st.toast(f"Error: {str(e)}", icon="❌")
        print(f"Error sending MQTT message: {e}")
        
        initialize_mqtt_client()

class SharedState:
    def __init__(self):
        self.latest_frame = None
        self.latest_frame_time = 0
        self.disease_info = "Waiting for detection results..." 
        self.is_disease_detected = False
        self.is_connected = False
        self.lock = threading.Lock()

state = SharedState()

def get_image_from_esp():
    try:
        response = requests.get(f"{ESP_CAM_URL}/capture", timeout=5)
        
        if response.status_code != 200 or len(response.content) < 100:
            index_response = requests.get(ESP_CAM_URL, timeout=5)
            
            response = requests.get(f"{ESP_CAM_URL}/stream", timeout=5, stream=True)
            
            if response.status_code != 200:
                response = requests.get(f"{ESP_CAM_URL}/jpg", timeout=5)
        
        if response.status_code == 200 and len(response.content) > 100:
            try:
                image = Image.open(BytesIO(response.content))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error processing image: {e}")
                return None
        else:
            print(f"Error getting image: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to ESP32-CAM: {e}")
        return None

def process_frame(frame, model):
    if frame is None:
        return None, "No frame received", False
    
    results = model(frame)
    
    result_frame = frame.copy()
    
    disease_detected = False
    disease_info = "No plant diseases detected"
    
    if results:
        result = results[0]
        boxes = result.boxes
        
        detected_classes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            detected_classes.append((class_name, confidence))
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{class_name} {confidence:.2f}"
            
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            cv2.rectangle(result_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
            
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        if detected_classes:
            for class_name, confidence in detected_classes:
                if "healthy" not in class_name.lower():
                    disease_detected = True
            
            if disease_detected:
                disease_info = "⚠️ Plant disease detected: " + ", ".join([f"{c[0]} ({c[1]:.2f})" for c in detected_classes])
            else:
                disease_info = "✅ Plants are healthy"
    
    result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    return result_frame_rgb, disease_info, disease_detected

def background_processing():
    model = load_model()
    print("Model loaded in background thread.")
    
    while True:
        try:
            frame = get_image_from_esp()
            
            if frame is not None:
                result_frame, disease_info, disease_detected = process_frame(frame, model)
                
                with state.lock:
                    state.latest_frame = result_frame
                    state.latest_frame_time = time.time()
                    state.disease_info = disease_info
                    state.is_disease_detected = disease_detected
                    state.is_connected = True
            else:
                with state.lock:
                    if time.time() - state.latest_frame_time > 10:
                        state.is_connected = False
            
            time.sleep(3)
            
        except Exception as e:
            print(f"Error in background thread: {e}")
            time.sleep(5)

def main():
    st.title("🌱 GrowNet Dashboard")
    st.markdown("### Real-time Plant Disease Detection System")
    
    with st.sidebar:
        st.subheader("System Information")
        connection_status = st.empty()
        last_update = st.empty()
        model_info = st.empty()
        
        st.subheader("Settings")
        refresh_rate = st.slider("UI Refresh Rate (seconds)", 1, 10, 2)
        
        model_info.info("Using YOLOv8 custom model for plant disease detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        camera_placeholder = st.empty()
    
    with col2:
        st.subheader("🔍 Detection Results")
        detection_placeholder = st.empty()
        
        st.subheader("ℹ️ Plant Health")
        health_status = st.empty()
        
        st.subheader("🔄 Control")
        if st.button("Activate Motor", use_container_width=True, type="primary"):
            send_mqtt_message()
    
    processing_thread = threading.Thread(target=background_processing, daemon=True)
    processing_thread.start()
    
    while True:
        with state.lock:
            frame = state.latest_frame
            disease_info = state.disease_info
            is_disease_detected = state.is_disease_detected
            is_connected = state.is_connected
            last_update_time = state.latest_frame_time
        
        if is_connected:
            connection_status.success("✅ Connected to ESP32-CAM")
            time_since_update = time.time() - last_update_time
            last_update.info(f"Last update: {int(time_since_update)} seconds ago")
        else:
            connection_status.error("❌ Not connected to ESP32-CAM")
            camera_placeholder.error("Cannot connect to camera feed. Check ESP32-CAM connection.")
        
        if frame is not None:
            camera_placeholder.image(frame, caption="ESP32-CAM Feed", use_container_width=True)
            
            if is_disease_detected:
                detection_placeholder.error(disease_info)
                health_status.warning("Plants need attention! Disease detected.")
            else:
                detection_placeholder.success(disease_info)
                health_status.success("All plants appear healthy.")
        
        time.sleep(refresh_rate)

if __name__ == "__main__":
    initialize_mqtt_client()
    main()