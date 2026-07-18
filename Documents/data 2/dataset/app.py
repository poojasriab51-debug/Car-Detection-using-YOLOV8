import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1. Page Configuration & Setup
st.set_page_config(
    page_title="AI Car Detection Dashboard",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 AI Car Detection Dashboard")
st.caption("Powered by YOLOv8 & Streamlit")
st.write("---")

# 2. Load the Custom YOLO Model safely
@st.cache_resource
def load_yolo_model():
    # Looks for best.pt directly in the folder it runs from
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error("⚠️ Could not find 'best.pt' right next to this file. Make sure you copied it successfully.")
    st.stop()

# 3. Sidebar Configuration
st.sidebar.header("🔧 Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05
)

# 4. Image Upload Widget
uploaded_file = st.file_uploader("Choose an image to scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Visual Layout: Before and After side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Detections")
        with st.spinner("Running YOLOv8 inference..."):
            # Run inference using the custom model
            results = model.predict(source=image, conf=confidence)
            
            # Plot boxes (convert BGR output to RGB for Streamlit display)
            res_plotted = results[0].plot()
            detected_image = Image.fromarray(res_plotted[..., ::-1])
            
            st.image(detected_image, use_container_width=True)
            
            # Metrics display below the image
            car_count = len(results[0].boxes)
            st.metric(label="Vehicles Detected", value=car_count)