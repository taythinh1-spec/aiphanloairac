import base64
import re
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Khởi tạo biến app độc lập ngay tại đây để loại bỏ lỗi ImportError trên Render
app = Flask(__name__, template_folder='templates')

# Tải mô hình YOLOv8 Nano siêu nhẹ
model_ai = YOLO('yolov8n.pt')

# RAM database tạm thời để lưu tổng điểm thi đua học sinh
USER_POINTS_DB = {}

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Không nhận được dữ liệu ảnh!'}), 400
        
        username = data.get('username', 'Học sinh ẩn danh').strip()
        if not username:
            username = 'Học sinh ẩn danh'

        # 1. Giải mã Base64 sang ảnh OpenCV OpenCV
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        img_bytes = base64.b64decode(image_data)
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'error': 'Dữ liệu hình ảnh bị hỏng!'}), 400

        # 2. Đưa ảnh vào mô hình YOLOv8 quét vật thể
        results = model_ai(frame)
        
        highest_conf = 0
        loai_rac = "Vật thể lạ / Rác vô cơ khác"
        huong_dan = "RÁC CÒN LẠI: Nếu đây là túi nilon, hộp xốp bẩn hoặc khăn giấy cũ, hãy vứt vào THÙNG RÁC VÔ CƠ (Màu xám/vàng) nhé!"
        diem_cong = 2

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                original_name = model_ai.names[cls]

                if conf > highest_conf and conf > 0.35:
                    highest_conf = conf
                    
                    # Phân loại rác tái chế
                    if original_name in ['bottle', 'cup', 'can', 'wine glass']:
                        loai_rac = "Chai lọ nhựa / Lon nhôm tái chế"
                        huong_dan = "🍀 RÁC TÁI CHẾ: Vui lòng súc sạch nước tồn đọng bên trong, ép dẹp (nếu được) và bỏ vào THÙNG RÁC MÀU XANH LÁ của trường để tích điểm cao nhé!"
                        diem_cong = 10
                    elif original_name in ['book', 'paper']:
                        loai_rac = "Sách báo / Giấy vụn"
                        huong_dan = "🍀 RÁC TÁI CHẾ: Hãy vuốt phẳng, xếp gọn gàng tránh làm ướt bẩn và để vào KHU VỰC THU GOM GIẤY VỤN thi đua kế hoạch nhỏ của lớp."
                        diem_cong = 10
                    
                    # Phân loại rác hữu cơ
                    elif original_name in ['apple', 'banana', 'orange', 'sandwich', 'cake', 'broccoli']:
                        loai_rac = "Thức ăn thừa / Vỏ trái cây"
                        huong_dan = "🍌 RÁC HỮU CƠ: Bạn đổ phần thức ăn thừa hoặc vỏ cây này vào THÙNG RÁC MÀU XANH DƯƠNG chuyên dụng để nhà trường ủ làm phân bón cây xanh."
                        diem_cong = 5

                    # Phân loại thiết bị nguy hiểm / rác điện tử
                    elif original_name in ['cell phone', 'laptop', 'remote', 'keyboard', 'mouse', 'scissors']:
                        loai_rac = "Thiết bị điện tử / Vật sắc nhọn"
                        huong_dan = "⚠️ RÁC NGUY HIỂM: Tuyệt đối không vứt chung vào thùng rác sinh hoạt. Hãy mang đến THÙNG THU GOM PIN VÀ ĐIỆN TỬ CŨ tại văn phòng Đoàn trường để xử lý riêng."
                        diem_cong = 20

        chuoi_hien_thi = f"📍 ĐỒ VẬT: {loai_rac.upper()}\n💡 GIẢI PHÁP: {huong_dan}"

        # 3. Lưu dữ liệu điểm thi đua
        if username not in USER_POINTS_DB:
            USER_POINTS_DB[username] = 0
        USER_POINTS_DB[username] += diem_cong

        return jsonify({
            'success': True,
            'prediction': chuoi_hien_thi,
            'diem_cong_tu_ai': diem_cong,
            'tong_diem_he_thong': USER_POINTS_DB[username]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
