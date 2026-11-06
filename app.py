from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

# نماذج جاهزة (pre-trained) من Hugging Face — تتحمّل تلقائياً أول مرة.
# لتغيير نموذج: بدّل الـ ID هنا فقط.
MODELS = {
    "chest": "lxyuan/vit-xray-pneumonia-classification",
    "brain": "Hemg/Brain-Tumor-Classification",
}

# ترجمة أسماء النتائج للعربي (أي اسم مش موجود هنا بيظهر زي ما هو)
AR = {
    "pneumonia": "التهاب رئوي (Pneumonia)",
    "normal": "سليمة",
    "notumor": "سليمة (لا يوجد ورم)",
    "glioma": "ورم دبقي (Glioma)",
    "meningioma": "ورم سحائي (Meningioma)",
    "pituitary": "ورم في الغدة النخامية (Pituitary)",
    "healthy": "سليمة",
}
NORMAL_WORDS = ("normal", "healthy", "no tumor", "notumor", "no finding", "negative")
LOW_CONFIDENCE = 0.70

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
_pipes = {}


def get_pipe(kind):
    if kind not in _pipes:
        _pipes[kind] = pipeline("image-classification", model=MODELS[kind])
    return _pipes[kind]


def clean(label):
    return label.lower().replace("_", " ").replace("-", " ").strip()


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/analyze")
def analyze():
    kind, f = request.form.get("kind"), request.files.get("image")
    if kind not in MODELS or not f:
        return jsonify(error="ارفع صورة واختر نوع الأشعة"), 400
    try:
        img = Image.open(f.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="الملف ليس صورة صالحة (استخدم JPG أو PNG)"), 400

    results = get_pipe(kind)(img, top_k=3)
    top = results[0]
    ar = lambda l: AR.get(clean(l).replace(" ", ""), l)
    return jsonify(
        label=top["label"],
        label_ar=ar(top["label"]),
        normal=any(w in clean(top["label"]) for w in NORMAL_WORDS),
        confidence=round(top["score"] * 100, 1),
        low_confidence=top["score"] < LOW_CONFIDENCE,
        top=[{"label_ar": ar(r["label"]), "score": round(r["score"] * 100, 1)} for r in results],
    )


if __name__ == "__main__":
    for k in MODELS:  # تحميل النماذج عند التشغيل عشان أول طلب ما يتأخرش
        print(f"Loading {k} model...")
        get_pipe(k)
    app.run(host="127.0.0.1", port=5000)
