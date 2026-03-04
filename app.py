import streamlit as st
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pickle
from datetime import datetime

st.set_page_config(page_title="AI Lost & Found", page_icon="🔍", layout="wide")
st.title("🔍 AI-Powered Lost & Found System")
st.markdown("**Upload photo → AI finds matches instantly** | Everything saved on your laptop")

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_clip_model()

# EMBEDDING + RGBA
def get_image_embedding(image):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        image_embeds = vision_outputs.pooler_output
        features = model.visual_projection(image_embeds)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].cpu().numpy()

DB_FILE = "lost_found_db.pkl"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f: return pickle.load(f)
    return {"lost": [], "found": []}

def save_db(db):
    with open(DB_FILE, "wb") as f: pickle.dump(db, f)

def find_matches(new_embedding, opposite_list, threshold=0.75):
    matches = []
    for item in opposite_list:
        sim = float(np.dot(new_embedding, item["embedding"]))
        if sim >= threshold:
            matches.append((item, sim))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:5]

tab1, tab2, tab3, tab4 = st.tabs(["📌 Report Lost", "📌 Report Found", "🔎 View All", "📊 Stats"])
db = load_db()

# ====================== REPORT LOST ======================
with tab1:
    st.header("Report Lost Item")
    with st.form("lost_form", clear_on_submit=True):
        desc = st.text_input("Description (e.g. Red AirPods)")
        location = st.text_input("Last seen location")
        contact = st.text_input("Your phone/email")
        file = st.file_uploader("Upload photo", type=["jpg","jpeg","png"])
        submitted = st.form_submit_button("🚨 Report as LOST")
        
        if submitted and file and desc and contact:
            image = Image.open(file).convert("RGB")    
            emb = get_image_embedding(image)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(UPLOAD_FOLDER, f"lost_{ts}.jpg")
            image.save(path)
            db["lost"].append({"id": len(db["lost"])+1, "desc":desc, "location":location, "contact":contact, "image_path":path, "embedding":emb, "timestamp":datetime.now().isoformat()})
            save_db(db)
            st.success("✅ Lost item reported!")
            matches = find_matches(emb, db["found"])
            if matches:
                st.subheader("🎉 AI Found Matches!")
                for m, s in matches:
                    c1, c2 = st.columns([1, 3])
                    with c1: st.image(m["image_path"], width=150)
                    with c2: st.write(f"**{m['desc']}** — Similarity: {s:.1%} — Contact: {m['contact']}")

#           REPORT FOUND          
with tab2:
    st.header("Report Found Item")
    with st.form("found_form", clear_on_submit=True):
        desc = st.text_input("Description (e.g. Blue wallet)")
        location = st.text_input("Where you found it")
        contact = st.text_input("Your phone/email")
        file = st.file_uploader("Upload photo", type=["jpg","jpeg","png"])
        submitted = st.form_submit_button("📍 Report as FOUND")
        
        if submitted and file and desc and contact:
            image = Image.open(file).convert("RGB")          # <-- FIXED RGBA error
            emb = get_image_embedding(image)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(UPLOAD_FOLDER, f"found_{ts}.jpg")
            image.save(path)
            db["found"].append({"id": len(db["found"])+1, "desc":desc, "location":location, "contact":contact, "image_path":path, "embedding":emb, "timestamp":datetime.now().isoformat()})
            save_db(db)
            st.success("✅ Found item reported!")
            matches = find_matches(emb, db["lost"])
            if matches:
                st.subheader("🎉 AI Found Owners!")
                for m, s in matches:
                    c1, c2 = st.columns([1, 3])
                    with c1: st.image(m["image_path"], width=150)
                    with c2: st.write(f"**{m['desc']}** — Similarity: {s:.1%} — Contact: {m['contact']}")

#            VIEW ALL & STATS 
with tab3:
    st.header("All Items")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Lost Items ({len(db['lost'])})")
        for i in reversed(db["lost"]):
            with st.expander(f"#{i['id']} {i['desc'][:50]}..."):
                st.image(i["image_path"], width=300)
                st.write(f"Location: {i['location']}")
                st.write(f"Contact: {i['contact']}")
    with c2:
        st.subheader(f"Found Items ({len(db['found'])})")
        for i in reversed(db["found"]):
            with st.expander(f"#{i['id']} {i['desc'][:50]}..."):
                st.image(i["image_path"], width=300)
                st.write(f"Location: {i['location']}")
                st.write(f"Contact: {i['contact']}")

with tab4:
    st.header("Stats")
    st.metric("Lost Items", len(db["lost"]))
    st.metric("Found Items", len(db["found"]))


st.caption("AI Lost & Found • CLIP vision matching • 100% local")
