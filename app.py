import streamlit as st
import numpy as np
import cv2
import pickle
import faiss
import insightface

# 🚀 모델 준비
@st.cache_resource
def load_model():
    model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    model.prepare(ctx_id=0)
    return model

model = load_model()

# 🚀 FAISS 인덱스와 라벨 로드
@st.cache_resource
def load_faiss():
    index = faiss.read_index("faiss_index.index")
    with open("faiss_labels.pkl", "rb") as f:
        labels = pickle.load(f)
    return index, labels

index, labels = load_faiss()

# 🚀 얼굴 임베딩 추출 함수
def get_face_embedding(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img)
    if faces:
        return faces[0].embedding
    else:
        return None

# 🚀 Streamlit UI
st.title("🧑‍💻 얼굴 유사도 판별 (FAISS)")
st.image(
    ["anthony_joshua.png","kang_ho_dong.png",
     "karina.png","pak_myung_su.png","unganoo.png"],
    caption=["Anthony Joshua", "강호동", 
             "카리나", "박명수", "Francis Ngannou"],
    #use_column_width=True
    width=300
)



uploaded_file = st.file_uploader("얼굴 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="업로드한 이미지", use_column_width=True)

    if st.button("예측하기"):
        embedding = get_face_embedding(uploaded_file.read())

        if embedding is None:
            st.error("❌ 얼굴을 감지하지 못했습니다.")
        else:
            emb = embedding.reshape(1, -1).astype("float32")
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)  # ⭐️ 정규화
            D, I = index.search(emb, k=1)
            best_idx = I[0][0]
            best_score = D[0][0]
            predicted_label = labels[best_idx]
            st.success(f"✅ 예측 결과: **{predicted_label}** (유사도: {best_score:.4f})")
