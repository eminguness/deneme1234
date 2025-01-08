from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from flask_cors import CORS
import os
import subprocess
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
CORS(app)

# Çıktı dizini
OUTPUT_FOLDER = 'runs/detect'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB Dosya Limiti

# Ana sayfa (ilk açılışta yönlendirme)
@app.route('/')
def home():
    return render_template('anasayfa.html')

# Analiz sayfası
@app.route('/analiz')
def analiz():
    return render_template('index.html')

# Hakkımızda sayfası
@app.route('/hakkimizda')
def hakkimizda():
    return render_template('hakkimizda.html')

# İletişim sayfası
@app.route('/iletisim')
def iletisim():
    return render_template('iletisim.html')

# Resim yükleme ve işleme
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Dosya seçilmedi"}), 400

    # Yüklenen resmi geçici dosyaya kaydet
    temp_image_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(file.filename))
    image = Image.open(file.stream)
    image.save(temp_image_path)

    # YOLOv7 modelini çalıştır
    weights_path = "C:/Users/Emin/Desktop/yolov7-catlak-siniflandirma-staj/weights/best.pt"
    detect_script = "C:/Users/Emin/Desktop/yolov7-catlak-siniflandirma-staj/yolov7/detect.py"

    try:
        result = subprocess.run([
            "python", detect_script,
            "--weights", weights_path,
            "--img", "640",
            "--conf", "0.25",
            "--source", temp_image_path,
            "--project", app.config['OUTPUT_FOLDER'],
            "--name", "exp",
            "--exist-ok"
        ], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Model çalıştırılamadı: {str(e)}")
        return jsonify({"error": f"Model çalıştırılamadı: {str(e)}"}), 500

    # En son çıkan görüntü klasörünü bul
    all_dirs = [d for d in os.listdir(app.config['OUTPUT_FOLDER']) if os.path.isdir(os.path.join(app.config['OUTPUT_FOLDER'], d))]
    if not all_dirs:
        return jsonify({"error": "Çıktı dizini bulunamadı"}), 500

    latest_exp_dir = sorted(all_dirs, key=lambda x: int(x.replace("exp", "")) if x.replace("exp", "").isdigit() else -1, reverse=True)[0]
    result_img_path = os.path.join(app.config['OUTPUT_FOLDER'], latest_exp_dir, file.filename)

    if os.path.exists(result_img_path):
        result_url = url_for('serve_detected_image', filename=f"{latest_exp_dir}/{file.filename}")
        return jsonify({"message": "Analiz tamamlandı!", "result_image_url": result_url}), 200
    else:
        return jsonify({"error": "Sonuç resmi bulunamadı"}), 500

# İşlenmiş resimleri servis etme
@app.route('/runs/detect/<path:filename>')
def serve_detected_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
