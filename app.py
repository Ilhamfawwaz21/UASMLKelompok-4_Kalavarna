from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import pandas as pd
from werkzeug.utils import secure_filename
from preproGambar import skin, eye
from analyze import predict

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB maximum upload size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('analyze.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('analyze.html', error='No selected file')
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and get results
            result_skin = skin(file_path)
            result_eye = eye(file_path)
            # Add RGB values to the result
            R_skin = result_skin[0]
            G_skin = result_skin[1]
            B_skin = result_skin[2]
            R_eye = result_eye[0]
            G_eye = result_eye[1]
            B_eye = result_eye[2]
            data = pd.DataFrame({
                'Skin_R': [R_skin],
                'Skin_G': [G_skin],
                'Skin_B': [B_skin],
                'Eye_R': [R_eye],
                'Eye_G': [G_eye],
                'Eye_B': [B_eye]
            })
            result = predict(data)
        return render_template('analyze.html', result=result, image_path=filename)
    return render_template('analyze.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)