import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import torch
import streamlit as st
import zipfile
import faiss
import json
import numpy as np
import cv2
from PIL import Image
import tempfile

from transformers import AutoImageProcessor, AutoModelForObjectDetection, CLIPProcessor, CLIPModel

# -------------------- Config --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_ckpt = "valentinafeve/yolos-fashionpedia"

CATS = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
    'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
    'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer',
    'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood',
    'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper',
    'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'
]

custom_label_map = {
    "shirt, blouse": "top", "top, t-shirt, sweatshirt": "top", "sweater": "top",
    "cardigan": "top", "vest": "top", "pants": "bottom", "shorts": "bottom",
    "skirt": "bottom", "dress": "dress", "jumpsuit": "dress", "jacket": "jacket",
    "coat": "jacket", "shoe": "shoes", "bag, wallet": "bag", "watch": "accessory",
    "glasses": "accessory", "hat": "accessory", "headband, head covering, hair accessory": "accessory"
}

# -------------------- Ensure Catalog is Extracted --------------------
@st.cache_resource
def ensure_catalog_folder():
    if not os.path.exists("catalog_crops"):
        if os.path.exists("catalog_crops.zip"):
            with zipfile.ZipFile("catalog_crops.zip", "r") as zip_ref:
                zip_ref.extractall("catalog_crops")
            st.success("catalog_crops.zip extracted successfully.")
        else:
            st.warning("catalog_crops.zip not found.")
    return "catalog_crops"

ensure_catalog_folder()

# -------------------- Load Models --------------------
@st.cache_resource
def load_models():
    det_processor = AutoImageProcessor.from_pretrained(det_ckpt)
    det_model = AutoModelForObjectDetection.from_pretrained(det_ckpt).to(device).eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.load_state_dict(torch.load("clip_model.pth", map_location=device))

    index = faiss.read_index("faiss_catalog.index")
    with open("metadata_fixed.json", "r") as f:
        metadata = json.load(f)

    return det_processor, det_model, clip_processor, clip_model, index, metadata

det_processor, det_model, clip_processor, clip_model, index, catalog_metadata = load_models()

# -------------------- Utility --------------------
def extract_frames_from_video(video_path, interval=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        i += 1
    cap.release()
    return frames

def detect_and_match(image):
    results = []
    inputs = det_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = det_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    detections = det_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

    for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        label_name = CATS[label.item()]
        if label_name not in custom_label_map:
            continue
        class_label = custom_label_map[label_name]
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image.crop((x1, y1, x2, y2))

        clip_inputs = clip_processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**clip_inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        emb_np = emb.cpu().numpy().astype('float32')

        D, I = index.search(emb_np, k=1)
        sim_score = 1 - D[0][0]
        top_match = catalog_metadata[I[0][0]]

        match_path = os.path.join("catalog_crops", top_match["image_path"])

        result = {
            "label": class_label,
            "score": float(score),
            "sim_score": float(sim_score),
            "crop": crop,
            "match_path": match_path,
            "match_id": top_match["product_id"]
        }
        results.append(result)
    return results

# -------------------- Streamlit UI --------------------
st.title("🔍 AI Fashion Search from Video")

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
frame_interval = st.slider("Frame Interval", 1, 60, 30, help="Extract 1 frame every N frames")

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    with st.spinner("Extracting frames..."):
        frames = extract_frames_from_video(video_path, interval=frame_interval)

    st.success(f"{len(frames)} frames extracted. Processing...")

    for idx, frame in enumerate(frames):
        st.subheader(f"🖼 Frame {idx + 1}")
        with st.spinner("Detecting and matching..."):
            results = detect_and_match(frame)

        if not results:
            st.info("No fashion items detected.")
            continue

        for result in results:
            col1, col2 = st.columns(2)
            with col1:
                st.image(result["crop"], caption=f"Detected: {result['label']} (score: {result['score']:.2f})")
            with col2:
                # Fix the path to point to actual location
                catalog_root = "catalog_crops/content/catalog_crops"
                filename = os.path.basename(result["match_path"])
                fixed_path = os.path.join(catalog_root, filename)
        
                if os.path.exists(fixed_path):
                    st.image(fixed_path, caption=f"Match ID: {result['match_id']} (sim: {result['sim_score']:.3f})")
                else:
                    st.warning(f"Image not found:\n{fixed_path}")



# # All real image paths from disk
# real_paths = {}
# for root, dirs, files in os.walk("catalog_crops"):
#     for name in files:
#         if name.endswith(".jpg"):
#             real_paths[name] = os.path.join(root, name)

# # Suppose your original path says 'catalog_crops/catalog_crops/15102_top_2.jpg'
# broken_paths = [
#     'catalog_crops/catalog_crops/15102_top_2.jpg',
#     'catalog_crops/catalog_crops/14976_dress_1.jpg'
# ]

# # Try to fix them by filename lookup
# fixed_paths = []
# for bp in broken_paths:
#     fname = os.path.basename(bp)
#     if fname in real_paths:
#         fixed_paths.append(real_paths[fname])
#     else:
#         print(f"❌ File not found: {fname}")

# print("✅ Fixed paths:", fixed_paths)


# import json
# import os

# # Load metadata
# with open("metadata.json", "r") as f:
#     metadata = json.load(f)

# # Fix each path
# for item in metadata:
#     image_filename = os.path.basename(item["image_path"])
#     corrected_path = os.path.join("catalog_crops", "content", "catalog_crops", image_filename)
#     item["image_path"] = corrected_path.replace("\\", "/")  # normalize for all OS

# # Save corrected metadata
# with open("metadata_fixed.json", "w") as f:
#     json.dump(metadata, f, indent=2)

# print("✅ metadata_fixed.json written with corrected paths.")
