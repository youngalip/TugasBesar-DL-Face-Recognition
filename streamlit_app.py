import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="VGGFace Face Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E40AF;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #EF4444;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# LOAD MODELS & CONFIG
# ===========================
@st.cache_resource
def load_models():
    """Load VGGFace model, MTCNN detector, and class names"""
    try:
        # Load VGGFace model
        model = tf.keras.models.load_model("vgg_model.h5")
        
        # Load class names
        class_names = np.load("class_names.npy", allow_pickle=True)
        
        # Initialize MTCNN
        detector = MTCNN()
        
        # Load config
        with open("vgg_config.json", "r") as f:
            config = json.load(f)
        
        return model, detector, class_names, config
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# ===========================
# FACE DETECTION & PREDICTION
# ===========================
def detect_and_predict(image, model, detector, class_names, image_size=224, 
                       preprocess_version=2, threshold=0.5):
    """
    Detect face and predict identity
    
    Args:
        image: PIL Image or numpy array
        model: VGGFace model
        detector: MTCNN detector
        class_names: List of class names
        image_size: Input size for model
        preprocess_version: VGGFace preprocessing version
        threshold: Confidence threshold for unknown detection
    
    Returns:
        dict with prediction results
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_array
    
    # Detect faces
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) == 0:
        return {
            "success": False,
            "message": "No face detected in the image",
            "face_detected": False
        }
    
    # Get first face (largest)
    face_data = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = face_data["box"]
    x, y = abs(x), abs(y)
    confidence_detection = face_data["confidence"]
    
    # Crop face with bounds checking
    x2 = min(x + w, img_rgb.shape[1])
    y2 = min(y + h, img_rgb.shape[0])
    face = img_rgb[y:y2, x:x2]
    
    if face.size == 0:
        return {
            "success": False,
            "message": "Failed to crop face",
            "face_detected": True
        }
    
    # Resize face
    face_resized = cv2.resize(face, (image_size, image_size)).astype("float32")
    
    # Preprocess for VGGFace
    face_pp = preprocess_input(face_resized.copy(), version=preprocess_version)
    face_pp = np.expand_dims(face_pp, axis=0)
    
    # Predict
    predictions = model.predict(face_pp, verbose=0)[0]
    
    # Get top 5 predictions
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    top_5_names = [class_names[i] for i in top_5_idx]
    top_5_confidences = predictions[top_5_idx]
    
    # Best prediction
    predicted_idx = top_5_idx[0]
    predicted_name = class_names[predicted_idx]
    predicted_confidence = predictions[predicted_idx]
    
    # Unknown detection
    is_unknown = predicted_confidence < threshold
    
    return {
        "success": True,
        "face_detected": True,
        "is_unknown": is_unknown,
        "predicted_name": predicted_name if not is_unknown else "UNKNOWN",
        "confidence": float(predicted_confidence),
        "top_5_names": top_5_names,
        "top_5_confidences": top_5_confidences.tolist(),
        "face_image": face_resized,
        "face_bbox": (x, y, w, h),
        "detection_confidence": float(confidence_detection),
        "original_image": img_rgb
    }

# ===========================
# VISUALIZATION
# ===========================
def plot_top_predictions(top_names, top_confidences, threshold, is_unknown):
    """Plot top 5 predictions as bar chart"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#EF4444' if is_unknown else '#10B981' if i == 0 else '#3B82F6' 
              for i in range(len(top_names))]
    
    bars = ax.barh(top_names, top_confidences, color=colors, alpha=0.8)
    
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (name, conf) in enumerate(zip(top_names, top_confidences)):
        ax.text(conf + 0.02, i, f'{conf:.2%}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def draw_face_box(image, bbox, label, confidence, color=(0, 255, 0)):
    """Draw bounding box on face"""
    img_copy = image.copy()
    x, y, w, h = bbox
    
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)
    
    # Draw label background
    label_text = f"{label}: {confidence:.1%}"
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_copy, (x, y - text_height - 10), (x + text_width, y), color, -1)
    cv2.putText(img_copy, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img_copy

# ===========================
# MAIN APP
# ===========================
def main():
    # Header
    st.markdown('<p class="main-header">👤 VGGFace Face Recognition System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to identify the person</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading models..."):
        model, detector, class_names, config = load_models()
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence to recognize a person. Below this = UNKNOWN"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        <div class="info-box">
        <b>Model:</b> VGGFace ResNet50<br>
        <b>Classes:</b> {len(class_names)}<br>
        <b>Input Size:</b> {config['image_size']}×{config['image_size']}<br>
        <b>Preprocessing:</b> Version {config['version']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("📋 Class List"):
            for i, name in enumerate(class_names, 1):
                st.text(f"{i}. {name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear face photo"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Recognize Face", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing..."):
                    result = detect_and_predict(
                        image, model, detector, class_names,
                        image_size=config['image_size'],
                        preprocess_version=config['version'],
                        threshold=threshold
                    )
                
                # Store result in session state
                st.session_state['result'] = result
    
    with col2:
        st.subheader("📊 Results")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            
            if not result['success']:
                st.markdown(f"""
                <div class="error-box">
                ❌ <b>{result['message']}</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show prediction
                if result['is_unknown']:
                    st.markdown(f"""
                    <div class="error-box">
                    ⚠️ <b>UNKNOWN PERSON</b><br>
                    Confidence: {result['confidence']:.2%} (below threshold: {threshold:.0%})
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ <b>Recognized: {result['predicted_name']}</b><br>
                    Confidence: {result['confidence']:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                metric_cols = st.columns(3)
                with metric_cols[0]:
                    st.metric("Face Detection", f"{result['detection_confidence']:.1%}")
                with metric_cols[1]:
                    st.metric("Recognition", f"{result['confidence']:.1%}")
                with metric_cols[2]:
                    st.metric("Status", "✅ Known" if not result['is_unknown'] else "⚠️ Unknown")
                
                # Show cropped face
                st.markdown("#### 👤 Detected Face")
                face_img_display = cv2.cvtColor(result['face_image'].astype(np.uint8), cv2.COLOR_RGB2BGR)
                st.image(face_img_display, caption="Cropped Face", width=200)
                
                # Show image with bounding box
                st.markdown("#### 🎯 Detection Result")
                color = (255, 0, 0) if result['is_unknown'] else (0, 255, 0)
                img_with_box = draw_face_box(
                    result['original_image'],
                    result['face_bbox'],
                    result['predicted_name'],
                    result['confidence'],
                    color
                )
                st.image(img_with_box, caption="Face Detection", use_container_width=True)
                
                # Top 5 predictions
                st.markdown("#### 📈 Top 5 Predictions")
                fig = plot_top_predictions(
                    result['top_5_names'],
                    result['top_5_confidences'],
                    threshold,
                    result['is_unknown']
                )
                st.pyplot(fig)
                
                # Detailed predictions table
                with st.expander("📋 Detailed Predictions"):
                    pred_data = {
                        "Rank": [1, 2, 3, 4, 5],
                        "Name": result['top_5_names'],
                        "Confidence": [f"{c:.2%}" for c in result['top_5_confidences']]
                    }
                    st.table(pred_data)
        else:
            st.info("👆 Upload an image and click 'Recognize Face' to start")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280;">
    <p><b>VGGFace Face Recognition System</b></p>
    <p>Powered by VGGFace ResNet50 + MTCNN | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()