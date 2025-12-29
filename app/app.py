import streamlit as st
import pickle
import re, regex
from underthesea import word_tokenize

# Load model
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model_svm.pkl", "rb"))

# Stopwords
VIETNAMESE_STOPWORDS = {
    "là","và","của","có","cho","một","những","các",
    "đã","đang","sẽ","này","đó","với","khi","tại",
    "theo","đến","từ","về","trong","ra","như"
}
NEGATION_WORDS = {"không","chưa","chẳng"}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = regex.sub(r"[^\p{L}\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, format="text").split()
    tokens = [w for w in tokens if (w not in VIETNAMESE_STOPWORDS) or (w in NEGATION_WORDS)]
    return " ".join(tokens)

# UI
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection")
st.write("Nhập nội dung bài báo tiếng Việt để kiểm tra")

text_input = st.text_area("📄 Nội dung bài báo", height=250)

if st.button("🔍 Kiểm tra"):
    if text_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung")
    else:
        clean = clean_text(text_input)
        X = tfidf.transform([clean])
        pred = model.predict(X)[0]

        if pred == 1:
            st.error("🚨 KẾT QUẢ: FAKE NEWS")
        else:
            st.success("✅ KẾT QUẢ: REAL NEWS")

        st.markdown("---")
        st.write("**Văn bản sau khi làm sạch:**")
        st.code(clean)
