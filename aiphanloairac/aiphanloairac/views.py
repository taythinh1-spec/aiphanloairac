import base64
import re
import cv2
import numpy as np
from flask import render_template, request, jsonify
from ultralytics import YOLO
from aiphanloairac import app  # Đảm bảo import đúng tên biến app của dự án

# Khởi tạo mô hình AI YOLOv8 phiên bản siêu nhẹ
model_ai = YOLO('yolov8n.pt')

# Database giả lập lưu trữ tổng quỹ điểm thi đua của học sinh trên RAM máy chủ
USER_POINTS_DB = {}

@app.route('/')
@app.route('/home')
def index():
    # Trả về giao diện trang chủ Cyberpunk Eco
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Không nhận được dữ liệu ảnh!'}), 400
        
        # Lấy tên học sinh, nếu để trống thì mặc định là Học sinh ẩn danh
        username = data.get('username', 'Học sinh ẩn danh').strip()
        if not username:
            username = 'Học sinh ẩn danh'

        # 1. GIẢI MÃ DỮ LIỆU ẢNH BASE64 GỬI TỪ TRÌNH DUYỆT ĐIỆN THOẠI
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        img_bytes = base64.b64decode(image_data)
        np_img = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'error': 'Dữ liệu hình ảnh bị hỏng hoặc lỗi định dạng!'}), 400

        # 2. ĐƯA ẢNH VÀO MÔ HÌNH AI YOLO ĐỂ PHÂN TÍCH VẬT THỂ
        results = model_ai(frame)
        
        highest_conf = 0
        loai_rac = "Vật thể lạ / Rác vô cơ khác"
        huong_dan = "RÁC CÒN LẠI: Nếu đây là túi nilon, hộp xốp bẩn hoặc khăn giấy cũ, hãy vứt vào THÙNG RÁC VÔ CƠ (Màu xám/vàng) nhé!"
        diem_cong = 2 # Điểm khuyến khích tối thiểu cho việc quét rác

        # Duyệt tìm vật thể AI quét được có độ tự tin cao nhất
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                original_name = model_ai.names[cls]

                # Chỉ lấy vật thể rõ ràng (Độ tự tin trên 35%) và cao nhất trong khung hình
                if conf > highest_conf and conf > 0.35:
                    highest_conf = conf
                    
                    # 3. LOGIC XỬ LÝ PHÂN LOẠI & GỢI Ý GIẢI PHÁP HỌC ĐƯỜNG
                    # Nhóm rác tái chế (Chai lọ, lon, sách báo...)
                    if original_name in ['bottle', 'cup', 'can', 'wine glass']:
                        loai_rac = "Chai lọ nhựa / Lon nhôm tái chế"
                        huong_dan = "🍀 RÁC TÁI CHẾ: Vui lòng súc sạch nước tồn đọng bên trong, ép dẹp (nếu được) và bỏ vào THÙNG RÁC MÀU XANH LÁ của trường để tích điểm cao nhé!"
                        diem_cong = 10
                    elif original_name in ['book', 'paper']:
                        loai_rac = "Sách báo / Giấy vụn"
                        huong_dan = "🍀 RÁC TÁI CHẾ: Hãy vuốt phẳng, xếp gọn gàng tránh làm ướt bẩn và để vào KHU VỰC THU GOM GIẤY VỤN thi đua kế hoạch nhỏ của lớp."
                        diem_cong = 10
                    
                    # Nhóm rác hữu cơ (Thức ăn vụn, trái cây...)
                    elif original_name in ['apple', 'banana', 'orange', 'sandwich', 'cake', 'broccoli']:
                        loai_rac = "Thức ăn thừa / Vỏ trái cây"
                        huong_dan = "🍌 RÁC HỮU CƠ: Bạn đổ phần thức ăn thừa hoặc vỏ cây này vào THÙNG RÁC MÀU XANH DƯƠNG chuyên dụng để nhà trường ủ làm phân bón cây xanh."
                        diem_cong = 5

                    # Nhóm rác điện tử / Nguy hiểm học đường
                    elif original_name in ['cell phone', 'laptop', 'remote', 'keyboard', 'mouse', 'scissors']:
                        loai_rac = "Thiết bị điện tử / Vật sắc nhọn"
                        huong_dan = "⚠️ RÁC NGUY HIỂM: Tuyệt đối không vứt chung vào thùng rác sinh hoạt. Hãy mang đến THÙNG THU GOM PIN VÀ ĐIỆN TỬ CŨ tại văn phòng Đoàn trường để xử lý riêng."
                        diem_cong = 20

        # 4. CHUẨN HÓA CHUỖI TEXT ĐỂ TRẢ VỀ KHỚP MÀN HÌNH HTML CYBERPUNK
        # Sử dụng ký tự \n để Javascript biên dịch xuống hàng chuẩn bằng hàm .replace()
        chuoi_hien_thi = f"📍 ĐỒ VẬT: {loai_rac.upper()}\n💡 GIẢI PHÁP: {huong_dan}"

        # 5. CỘNG VÀ LƯU QUỸ ĐIỂM THI ĐUA CHO HỌC SINH
        if username not in USER_POINTS_DB:
            USER_POINTS_DB[username] = 0
        USER_POINTS_DB[username] += diem_cong

        # 6. TRẢ DỮ LIỆU ĐỊNH DẠNG JSON VỀ TRÌNH DUYỆT
        return jsonify({
            'success': True,
            'prediction': chuoi_hien_thi,
            'diem_cong_tu_ai': diem_cong,
            'tong_diem_he_thong': USER_POINTS_DB[username]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
