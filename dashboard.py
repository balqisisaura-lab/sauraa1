import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_fixed.h5")  # ganti sesuai nama file model kamu
    return model

model = load_model()

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Dashboard Klasifikasi Gambar", page_icon="🧠", layout="centered")

st.title("🖼️ Dashboard Klasifikasi Gambar (Rock, Paper, Scissors)")
st.write("Upload gambar untuk diprediksi menggunakan model .h5 kamu.")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang di-upload
    image_uploaded = Image.open(uploaded_file)
    st.image(image_uploaded, caption="Gambar yang diunggah", use_column_width=True)

    # ==========================
    # Preprocessing
    # ==========================
    # Konversi ke grayscale karena model kamu dilatih pakai citra hitam-putih
    img = image_uploaded.convert('L')

    # Ubah ukuran sesuai input model (misal 128x128)
    img = img.resize((128, 128))

    # Ubah ke array numpy
    img_array = np.array(img)

    # Tambahkan dimensi batch & channel
    img_array = np.expand_dims(img_array, axis=-1)  # channel
    img_array = np.expand_dims(img_array, axis=0)   # batch

    # Normalisasi
    img_array = img_array / 255.0

    # ==========================
    # Prediksi
    # ==========================
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # Label kelas — ubah sesuai label dataset kamu
    class_names = ['Rock', 'Paper', 'Scissors']

    st.subheader("🔍 Hasil Prediksi")
    st.write(f"**Kelas:** {class_names[predicted_class]}")
    st.write(f"**Tingkat Kepercayaan:** {confidence:.2f}")

    # Tampilkan semua probabilitas
    st.bar_chart(predictions[0])

else:
    st.info("Silakan upload gambar untuk mulai prediksi.")

st.caption("💡 Pastikan gambar yang diunggah sesuai dengan data latih (grayscale, ukuran mirip).")
