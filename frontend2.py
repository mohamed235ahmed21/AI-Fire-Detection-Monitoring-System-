import streamlit as st
import requests
import cv2
import numpy as np
import base64
import os
import time
from PIL import Image
import io

# Set Streamlit page configurations
st.set_page_config(
    page_title="GuardianAI - Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Endpoint definition
API_URL = "http://127.0.0.1:8000"

# Custom CSS for Premium Styling (Dark Theme, Glassmorphism, Neon Orange Highlights)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Global font override */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main banner styling */
    .header-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(30,30,36,0.9) 0%, rgba(20,20,24,0.9) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 69, 0, 0.25);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(255, 69, 0, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(90deg, #FF4500 0%, #FF8C00 50%, #D21F3C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin-bottom: 0.2rem;
    }
    
    .header-subtitle {
        color: #A0A0A5;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Metrics Card styling */
    .card-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        flex: 1;
        background: rgba(30, 30, 36, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(255, 69, 0, 0.15);
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 69, 0, 0.4);
        box-shadow: 0 12px 24px rgba(255, 69, 0, 0.15);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #A0A0A5;
        letter-spacing: 0.1em;
    }
    
    /* Status indicators */
    .pulse-green {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #00E676;
        box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7);
        animation: pulse 1.5s infinite;
    }
    
    .pulse-red {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #FF1744;
        box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.7);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(0, 230, 118, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 230, 118, 0);
        }
    }
    
    /* Custom Alert panels */
    .alert-banner {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .alert-safe {
        background-color: rgba(0, 230, 118, 0.15);
        border: 1px solid rgba(0, 230, 118, 0.3);
        color: #00E676;
    }
    .alert-warning {
        background-color: rgba(255, 140, 0, 0.15);
        border: 1px solid rgba(255, 140, 0, 0.3);
        color: #FF8C00;
    }
    .alert-danger {
        background-color: rgba(255, 23, 68, 0.15);
        border: 1px solid rgba(255, 23, 68, 0.3);
        color: #FF1744;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to check API status
def check_backend_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None

# Fetch backend health
backend_online, health_data = check_backend_health()

# SIDEBAR: Navigation & Controls
st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 1rem;'>
        <h2>GuardianAI 🔥</h2>
        <p>Advanced Fire Detection HUD</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "NAVIGATION",
    ["📊 System Dashboard", "🖼️ Analyze Image", "📹 Analyze Video", "📷 Live Camera Feed"]
)

st.sidebar.markdown("---")

# Model Controls
st.sidebar.markdown("### MODEL CONFIGURATION")
conf_threshold = st.sidebar.slider(
    "Inference Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.05,
    help="Adjust threshold to balance precision and recall. Higher values limit false positives."
)

# Status indicators on Sidebar
st.sidebar.markdown("### SYSTEM STATUS")
if backend_online:
    model_name = health_data.get("model", "YOLOv8")
    st.sidebar.markdown(f"""
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span class='pulse-green'></span>
            <span style='color: #00E676; font-weight: bold;'>Backend API: Online</span>
        </div>
        <div style='margin-top: 5px; font-size: 0.85rem; color: #A0A0A5;'>
            <b>Model:</b> {model_name}<br>
            <b>Target Classes:</b> {list(health_data.get('classes', {}).values())}
        </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span class='pulse-red'></span>
            <span style='color: #FF1744; font-weight: bold;'>Backend API: Offline</span>
        </div>
        <div style='margin-top: 5px; font-size: 0.85rem; color: #A0A0A5;'>
            Please launch the FastAPI backend (app.py) on port 8000 to enable model inference.
        </div>
    """, unsafe_allow_html=True)

# MAIN HEADER BANNER
st.markdown("""
    <div class='header-container'>
        <div class='header-title'>GUARDIAN AI</div>
        <div class='header-subtitle'>Real-Time Computer Vision Fire & Thermal Hazard Detection System</div>
    </div>
""", unsafe_allow_html=True)

# PAGE 1: System Dashboard
if nav_selection == "📊 System Dashboard":
    # Show Banner image at the top of the dashboard
    if os.path.exists("banner.png"):
        st.image("banner.png", use_container_width=True)
    
    st.markdown("### SYSTEM OVERVIEW")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Core Detection Model</div>
                <div class='metric-val' style='color: #FF8C00;'>YOLOv8 Custom</div>
                <div style='font-size: 0.85rem; color: #A0A0A5;'>Trained on Roboflow Fire Dataset</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        api_status_color = "#00E676" if backend_online else "#FF1744"
        api_status_text = "HEALTHY" if backend_online else "DISCONNECTED"
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Inference Engine Status</div>
                <div class='metric-val' style='color: {api_status_color};'>{api_status_text}</div>
                <div style='font-size: 0.85rem; color: #A0A0A5;'>Listening on localhost:8000</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Target Anomalies</div>
                <div class='metric-val' style='color: #FF1744;'>Fire / Flame</div>
                <div style='font-size: 0.85rem; color: #A0A0A5;'>Single-class classification</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        #### Features:
        1. **High-Speed Object Detection**: Built using Ultralytics YOLOv8 for rapid inference on standard CPU/GPU hardware.
        2. **FastAPI Inference Hub**: A decoupled backend architecture that runs the model as an API endpoint, allowing flexible scaling.
        3. **Live Camera Feed**: Support for real-time camera captures directly in the browser.
        4. **Video Stream Analysis**: Processes and streams video frames with overlay indicators.
        
        #### How to use:
        * Choose **Analyze Image** from the sidebar to test fire detection on static files.
        * Choose **Analyze Video** to process and monitor video recordings.
        * Adjust the **Inference Confidence** threshold in the sidebar configuration to tweak detection sensitivity.
    """)

# PAGE 2: Analyze Image
elif nav_selection == "🖼️ Analyze Image":
    st.markdown("### 🖼️ Static Image Inference")
    st.markdown("Upload a photo or fire screenshot to run YOLOv8 object detection.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display layout cols
        col_img1, col_img2 = st.columns(2)
        
        # Read file bytes
        file_bytes = uploaded_file.read()
        
        with col_img1:
            st.markdown("#### Original Upload")
            st.image(file_bytes, use_container_width=True)
            
        with col_img2:
            st.markdown("#### Inference Results")
            
            if not backend_online:
                st.error("Cannot perform inference. Backend API is offline.")
            else:
                with st.spinner("Analyzing image for thermal anomalies..."):
                    # Prepare file payload
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    
                    try:
                        # Post request to FastAPI
                        response = requests.post(
                            f"{API_URL}/predict",
                            files=files,
                            params={"conf": conf_threshold},
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            count = data.get("count", 0)
                            detections = data.get("detections", [])
                            encoded_image = data.get("image", "")
                            
                            # Hazard Level alerts
                            if count == 0:
                                st.markdown("""
                                    <div class='alert-banner alert-safe'>
                                        <span>🛡️</span> Hazard Status: SAFE (No anomalies detected)
                                    </div>
                                """, unsafe_allow_html=True)
                            elif count <= 2:
                                st.markdown(f"""
                                    <div class='alert-banner alert-warning'>
                                        <span>⚠️</span> Hazard Status: WARNING ({count} fire zones detected)
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div class='alert-banner alert-danger'>
                                        <span>🚨</span> Hazard Status: CRITICAL DANGER ({count} fire zones detected!)
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Decode and render annotated image
                            img_data = base64.b64decode(encoded_image)
                            st.image(img_data, use_container_width=True)
                            
                            # Display individual detection statistics
                            if count > 0:
                                st.markdown("#### Detections Breakdown")
                                for i, det in enumerate(detections):
                                    st.write(
                                        f"**Zone {i+1}**: class `{det['class_name']}` "
                                        f"(Confidence: `{det['confidence']:.2%}`, Bbox: `{[round(c,1) for c in det['bbox']]}`)"
                                    )
                                    
                        else:
                            st.error(f"Error from API server: {response.text}")
                    except Exception as e:
                        st.error(f"Failed to communicate with API server: {e}")

# PAGE 3: Analyze Video
elif nav_selection == "📹 Analyze Video":
    st.markdown("### 📹 Video Inference Stream")
    st.markdown("Upload a video recording to process and detect fire hazards frame-by-frame.")
    
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        if not backend_online:
            st.error("Cannot perform inference. Backend API is offline.")
        else:
            # Save uploaded video to temp file for OpenCV to read
            temp_filename = "temp_uploaded_video.mp4"
            with open(temp_filename, "wb") as f:
                f.write(uploaded_video.read())
                
            # OpenCV capture
            cap = cv2.VideoCapture(temp_filename)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            st.markdown(f"**Video Stats**: {total_frames} frames, {fps:.1f} FPS")
            
            # Action button
            run_btn = st.button("🚀 Start Inference Pipeline")
            
            if run_btn:
                progress_bar = st.progress(0)
                status_txt = st.empty()
                frame_placeholder = st.empty()
                
                # Metrics indicators
                stats_col1, stats_col2 = st.columns(2)
                fire_frames_detected = 0
                frame_count = 0
                
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    # Send every 2nd or 3rd frame for performance, or all. Let's do all, but downscale slightly if needed.
                    # Downscale frame for faster network transfer
                    h, w, _ = frame.shape
                    if w > 640:
                        frame_resized = cv2.resize(frame, (640, int(h * (640 / w))))
                    else:
                        frame_resized = frame.copy()
                        
                    # Encode frame to jpg
                    _, encoded_frame = cv2.imencode('.jpg', frame_resized)
                    frame_bytes = encoded_frame.tobytes()
                    
                    # Post payload to FastAPI
                    try:
                        files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
                        response = requests.post(
                            f"{API_URL}/predict",
                            files=files,
                            params={"conf": conf_threshold},
                            timeout=5
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            count = data.get("count", 0)
                            encoded_image = data.get("image", "")
                            
                            if count > 0:
                                fire_frames_detected += 1
                                
                            # Update frame placeholder with annotated image
                            img_data = base64.b64decode(encoded_image)
                            frame_placeholder.image(img_data, channels="BGR", use_container_width=True)
                            
                        # Update progress
                        progress_val = min(1.0, frame_count / total_frames)
                        progress_bar.progress(progress_val)
                        status_txt.text(f"Processing frame {frame_count}/{total_frames}... (Fire alerts: {fire_frames_detected})")
                        
                    except Exception as e:
                        st.error(f"Error on frame {frame_count}: {e}")
                        break
                        
                cap.release()
                
                # Cleanup
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except Exception:
                        pass
                
                elapsed_time = time.time() - start_time
                st.success(f"Processing complete! Elapsed time: {elapsed_time:.1f}s. Fire detected in {fire_frames_detected} frames.")

# PAGE 4: Live Camera Feed
elif nav_selection == "📷 Live Camera Feed":
    st.markdown("### 📷 Live Camera Inference")
    st.markdown("Stream your webcam feed with real-time fire detection.")

    if not backend_online:
        st.error("Cannot perform inference. Backend API is offline. Please start backend1.py first.")
    else:
        col_feed, col_info = st.columns([2, 1])

        with col_info:
            st.markdown("#### Controls")
            start_btn = st.button("▶️ Start Live Detection", use_container_width=True,
                                  key="start_live")
            stop_btn  = st.button("⏹️ Stop", use_container_width=True,
                                  key="stop_live")
            st.markdown("---")
            alert_placeholder = st.empty()
            stats_placeholder = st.empty()
            alert_placeholder.markdown("""
                <div class='alert-banner alert-safe'>
                    <span>📷</span> Waiting for camera start...
                </div>
            """, unsafe_allow_html=True)

        with col_feed:
            frame_placeholder = st.empty()

        # Session state for camera
        if "live_running" not in st.session_state:
            st.session_state.live_running = False

        if stop_btn:
            st.session_state.live_running = False

        if start_btn:
            st.session_state.live_running = True

        if st.session_state.live_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam. Make sure a camera is connected.")
                st.session_state.live_running = False
            else:
                prev_time = time.time()
                total_fire_frames = 0
                total_frames = 0

                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Camera frame lost. Retrying...")
                        time.sleep(0.1)
                        continue

                    total_frames += 1

                    # Resize for faster transfer if needed
                    h, w, _ = frame.shape
                    if w > 640:
                        frame_resized = cv2.resize(frame, (640, int(h * (640 / w))))
                    else:
                        frame_resized = frame.copy()

                    # Encode frame to JPEG
                    _, encoded_frame = cv2.imencode('.jpg', frame_resized)
                    frame_bytes = encoded_frame.tobytes()

                    # Send to backend
                    try:
                        files = {"file": ("frame.jpg", frame_bytes, "image/jpeg")}
                        response = requests.post(
                            f"{API_URL}/predict",
                            files=files,
                            params={"conf": conf_threshold},
                            timeout=5
                        )

                        if response.status_code == 200:
                            data = response.json()
                            count = data.get("count", 0)
                            encoded_image = data.get("image", "")

                            # Calculate FPS
                            now = time.time()
                            fps = 1.0 / (now - prev_time + 1e-8)
                            prev_time = now

                            if count > 0:
                                total_fire_frames += 1

                            # Show annotated frame
                            img_data = base64.b64decode(encoded_image)
                            frame_placeholder.image(img_data, use_container_width=True)

                            # Update alert
                            if count == 0:
                                alert_placeholder.markdown("""
                                    <div class='alert-banner alert-safe'>
                                        <span>🛡️</span> Status: SAFE — No fire detected
                                    </div>
                                """, unsafe_allow_html=True)
                            elif count <= 2:
                                alert_placeholder.markdown(f"""
                                    <div class='alert-banner alert-warning'>
                                        <span>⚠️</span> WARNING: {count} fire zone(s) detected!
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                alert_placeholder.markdown(f"""
                                    <div class='alert-banner alert-danger'>
                                        <span>🚨</span> CRITICAL: {count} fire zone(s) detected!
                                    </div>
                                """, unsafe_allow_html=True)

                            # Update stats
                            stats_placeholder.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-title'>Live Stats</div>
                                    <div style='margin-top:0.5rem; text-align:left; font-size:0.9rem;'>
                                        <b>FPS:</b> {fps:.1f}<br>
                                        <b>Fires Now:</b> {count}<br>
                                        <b>Frame:</b> {total_frames}<br>
                                        <b>Fire Frames:</b> {total_fire_frames}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        # Show the raw frame if backend fails
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, use_container_width=True)
                        alert_placeholder.markdown(f"""
                            <div class='alert-banner alert-danger'>
                                <span>⚠️</span> Backend error: {e}
                            </div>
                        """, unsafe_allow_html=True)

                cap.release()
                st.session_state.live_running = False
                st.info("Camera stopped.")