import os
import cv2
import numpy as np
from flask import render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from aiphanloairac import app  # Đảm bảo đúng tên import dự án của cậu

# Khởi tạo mô hình AI YOLOv8
model_ai = YOLO('yolov8n.pt')

# Cấu hình thư mục tạm để lưu ảnh rác người dùng tải lên
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/upload_trash', methods=['POST'])
def upload_trash():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file gửi lên'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file ảnh'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Đọc ảnh bằng OpenCV để AI xử lý
        frame = cv2.imread(filepath)
        if frame is None:
            return jsonify({'error': 'Định dạng ảnh không hợp lệ'}), 400

        results = model_ai(frame)
        
        # Biến đếm xem phát hiện được bao nhiêu vật thể rác
        trash_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = int(box.conf[0] * 100)
                original_name = model_ai.names[cls]

                if conf > 35:  # Độ tự tin trên 35%
                    trash_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Phân loại rác bằng tư duy logic
                    label = f"{original_name}"
                    color = (255, 255, 255) # Trắng

                    if original_name in ['bottle', 'cup', 'can', 'book', 'wine glass']:
                        label = "RAC TAI CHE"
                        color = (0, 255, 0) # Xanh lá
                    elif original_name in ['apple', 'banana', 'orange', 'sandwich', 'cake']:
                        label = "RAC HUU CO"
                        color = (0, 165, 255) # Cam
                    elif original_name in ['cell phone', 'laptop', 'remote', 'keyboard', 'mouse']:
                        label = "RAC DIEN TU"
                        color = (0, 0, 255) # Đỏ

                    # Vẽ khung hình và chữ đè lên ảnh gốc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{label} {conf}%", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Lưu đè ảnh đã được vẽ khung AI lại vào thư mục static
        processed_filename = 'ai_' + filename
        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, frame)

        # Trả kết quả link ảnh về cho giao diện hiển thị mà không cần tải lại trang
        return jsonify({
            'success': True,
            'image_url': f'/static/uploads/{processed_filename}',
            'count': trash_count
        })
