import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

ARTIFACTS_DIR = "artifacts"
IMG_SIZE = 240
CONFIDENCE_THRESHOLD = 0.65  # Ambang batas untuk menghindari prediksi asal-asalan

KNOWLEDGE_BASE = {
    "Bacterial Spot": {
        "label": "Bercak Bakteri",
        "penanggulangan": "Gunakan agens hayati penginduksi resistensi seperti Bacillus subtilis (galur B01) dan Asam Salisilat (SA) untuk menekan tingkat keparahan infeksi. Bakterisida kimia juga dapat diaplikasikan jika infeksi sudah meluas.",
        "pencegahan": "Gunakan benih bersertifikat yang dijamin bebas patogen dan terapkan rotasi tanaman. Manajemen penyakit terpadu sejak fase pembibitan sangat penting untuk menghindari kerugian.",
        "alert": "error",
    },
    "Cercospora Leaf Spot": {
        "label": "Bercak Daun Cercospora",
        "penanggulangan": "Lakukan pemangkasan pada daun yang terinfeksi untuk memutus rantai spora jamur. Semprotkan fungisida secara berkala sejak gejala awal berupa bercak bulat cokelat kecil muncul di daun.",
        "pencegahan": "Terapkan manajemen lahan terpadu dengan menjaga sirikulasi udara. Keparahan penyakit ini meningkat drastis dari fase vegetatif ke reproduktif akibat tingginya curah hujan dan kelembapan.",
        "alert": "error",
    },
    "Curl Virus": {
        "label": "Virus Keriting Daun",
        "penanggulangan": "Karena tidak ada antivirus langsung, pengendalian difokuskan pada manajemen vektor kutu kebul (Bemisia tabaci). Gunakan perangkap kuning lengket (yellow sticky traps) dan aplikasikan insektisida sistemik.",
        "pencegahan": "Pasang jaring pelindung di area persemaian dan gunakan mulsa reflektif perak untuk mengusir hama pengisap. Pilih dan tanam varietas cabai yang memiliki ketahanan (resistan) terhadap virus ini.",
        "alert": "error",
    },
    "Healthy Leaf": {
        "label": "Daun Sehat",
        "penanggulangan": "Tidak diperlukan intervensi khusus. Tanaman dalam kondisi baik.",
        "pencegahan": "Pertahankan sanitasi kebun secara rutin dan gunakan benih bersertifikat. Lakukan pemantauan visual secara berkala untuk memastikan tidak ada gejala klorosis atau defisiensi hara.",
        "alert": "success",
    },
    "Nutrition Deficiency": {
        "label": "Defisiensi Nutrisi",
        "penanggulangan": "Berikan pemupukan berimbang yang presisi berdasarkan gejala visual spesifik (seperti klorosis) untuk mengatasi kekurangan unsur hara makro (N, P, K) maupun unsur mikro.",
        "pencegahan": "Lakukan deteksi dini berbasis pemantauan visual non-destruktif. Pastikan kebutuhan nutrisi dan ketersediaan unsur hara di dalam tanah selalu terpenuhi sebelum bibit ditanam.",
        "alert": "warning",
    },
    "White spot": {
        "label": "Bercak Putih",
        "penanggulangan": "Untuk infeksi embun tepung (Leveillula taurica), aplikasikan fungisida seperti Myclobutanil 10% WP atau Azoxystrobin. Sebagai alternatif ramah lingkungan, gunakan biopestisida kapang hiperparasit Ampelomyces quisqualis.",
        "pencegahan": "Kurangi tingkat kelembapan di sekitar kanopi dengan menjaga sirkulasi udara yang baik. Sangat disarankan untuk menanam kultivar cabai yang tahan terhadap penyakit embun tepung.",
        "alert": "warning",
    },
}

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, "chili_cnn_model.keras"))
    with open(os.path.join(ARTIFACTS_DIR, "class_names.json"), encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]
    return model, class_names

def preprocess_image(pil_image):
    """Preprocessing ketat agar identik dengan pipeline tf.data saat training."""
    img_array = np.array(pil_image.convert("RGB"))
    img_tensor = tf.image.resize(img_array, [IMG_SIZE, IMG_SIZE])
    img_tensor = tf.cast(img_tensor, tf.float32)
    return tf.expand_dims(img_tensor, axis=0)

def predict(img_tensor, model, class_names):
    probs = model.predict(img_tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Uncertain"
    else:
        predicted_class = class_names[idx]
        
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {c: float(p) for c, p in zip(class_names, probs)},
    }

def main():
    st.set_page_config(page_title="Deteksi Penyakit Daun Cabai", layout="centered")
    st.title("Deteksi Penyakit Berdasarkan Daun Cabai")

    model, class_names = load_artifacts()

    st.markdown("""
    ## Selamat Datang di Sistem Pendeteksi Penyakit Tanaman Cabai!
    Sistem ini memanfaatkan Deep Learning untuk mengidentifikasi berbagai penyakit tanaman cabai langsung dari gambar daunnya.
    """)
    st.divider()

    uploaded_file = st.file_uploader(
        "Pilih gambar daun cabai",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG",
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.image(pil_image, caption="Gambar yang diunggah", width="stretch")
        with col2:
            st.write("")
            st.write("")
            detect_clicked = st.button(
                "Deteksi Penyakit",
                type="primary",
                width="stretch",
            )

        if detect_clicked:
            try:
                with st.spinner("Menganalisis gambar..."):
                    img_tensor = preprocess_image(pil_image)
                    result = predict(img_tensor, model, class_names)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menganalisis gambar. Detail: {e}")
                st.stop()

            pred_class = result["predicted_class"]
            confidence = result["confidence"]
            
            st.divider()
            st.subheader("Hasil Prediksi")

            if pred_class == "Uncertain":
                st.warning(
                    f"Model tidak yakin dengan gambar ini (Confidence: {confidence:.1%}). "
                    "Pastikan gambar yang diunggah adalah objek daun cabai yang jelas dengan pencahayaan baik."
                )
            else:
                info = KNOWLEDGE_BASE.get(pred_class)
                if info is None:
                    st.error(f"Kelas terdeteksi (`{pred_class}`) tidak ada di KNOWLEDGE_BASE.")
                else:
                    alert_fn = {
                        "success": st.success,
                        "warning": st.warning,
                        "error": st.error,
                    }[info["alert"]]
                    
                    alert_fn(
                        f"Kondisi Tanaman: **{info['label']}** "
                        f"(Confidence: {confidence:.1%})"
                    )
                    
                    st.divider()
                    st.subheader("Informasi Penyakit")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Penanggulangan**")
                        st.info(info["penanggulangan"])
                    with col_b:
                        st.markdown("**Pencegahan**")
                        st.info(info["pencegahan"])

if __name__ == "__main__":
    main()