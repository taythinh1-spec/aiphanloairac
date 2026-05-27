import os
import json
import base64
import logging
import requests
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from datetime import datetime

# Cấu hình log để dễ theo dõi lỗi nếu có
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Flask App (Đảm bảo tên app trùng với cấu hình Gunicorn của bạn)
app = Flask(__name__)

# ==========================================
# 1. CẤU HÌNH KẾT NỐI MONGODB ATLAS
# ==========================================
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://YOUR_MONGO_URI_HERE")
try:
    client = MongoClient(MONGO_URI)
    db = client['AIPhanLoaiRac']       # Tên Database của bạn
    collection = db['LichSuQuet']      # Tên Collection lưu lịch sử
    logger.info("Đã kết nối MongoDB Atlas thành công.")
except Exception as e:
    logger.error(f"Lỗi kết nối MongoDB: {str(e)}")

# ==========================================
# 2. ĐƯỜNG DẪN TRANG CHỦ (GIAO DIỆN CHÍNH)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# 3. XỬ LÝ NHẬN ẢNH VÀ GỌI AI PHÂN LOẠI RÁC
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu ảnh Base64 từ giao diện gửi lên
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'Không nhận được dữ liệu ảnh từ Camera.'}), 400

        base64_image = data['image']
        
        # Xử lý chuỗi Base64: Lọc bỏ phần đầu định dạng nếu có (data:image/jpeg;base64,...)
        if "," in base64_image:
            image_data_clean = base64_image.split(",")[1]
        else:
            image_data_clean = base64_image

        # Lấy API Key từ biến môi trường trên Render
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("Chưa cấu hình GEMINI_API_KEY trong Environment Variables!")
            return jsonify({'success': False, 'error': 'Hệ thống chưa cấu hình API Key.'}), 500

        # Đường dẫn API trực tiếp lên máy chủ Google Gemini 1.5 Flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        # Chuẩn bị dữ liệu Payload gửi đi đúng chuẩn JSON của Google API
        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": (
                            "Bạn là một chuyên gia phân loại rác thông minh. Hãy nhìn vào bức ảnh này và thực hiện nhiệm vụ sau:\n"
                            "1. Xác định tên vật thể/loại rác xuất hiện trong ảnh.\n"
                            "2. Phân loại nó vào một trong ba nhóm chính xác: 'Rác hữu cơ', 'Rác vô cơ' hoặc 'Rác tái chế'.\n"
                            "3. Đưa ra hướng dẫn xử lý hoặc vứt bỏ ngắn gọn, thân thiện với môi trường.\n"
                            "Câu trả lời của bạn phải viết hoàn toàn bằng tiếng Việt, ngắn gọn, súc tích và dễ hiểu."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data_clean
                        }
                    }
                ]
            }]
        }
        
        headers = {'Content-Type': 'application/json'}
        
        # Gửi yêu cầu HTTP POST trực tiếp tới Google AI
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        # Kiểm tra phản hồi từ Google và trích xuất chuỗi văn bản kết quả
        if response.status_code == 200 and 'candidates' in response_data:
            ai_output = response_data['candidates'][0]['content']['parts'][0]['text']
            logger.info("AI phân tích ảnh thành công!")
            
            # --- LƯU LỊCH SỬ VÀO MONGODB ATLAS ---
            try:
                history_data = {
                    "ket_qua": ai_output,
                    "thoi_gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "hinh_anh": base64_image  # Lưu chuỗi ảnh để xem lại nếu muốn
                }
                collection.insert_one(history_data)
                logger.info("Đã lưu lịch sử quét vào MongoDB Atlas.")
            except Exception as mongo_err:
                logger.error(f"Không thể lưu vào MongoDB: {str(mongo_err)}")
                # Vẫn tiếp tục chạy dù lỗi database để không làm gián đoạn người dùng

            # Trả dữ liệu thành công về cho giao diện Javascript hiển thị lên màn hình
            return jsonify({'success': True, 'prediction': ai_output})
            
        else:
            logger.error(f"Lỗi API từ Google: {json.dumps(response_data)}")
            return jsonify({'success': False, 'error': 'Hệ thống AI bận: Không thể nhận diện ảnh vào lúc này.'}), 502

    except Exception as e:
        logger.error(f"Lỗi hệ thống nghiêm trọng tại hàm predict: {str(e)}")
        return jsonify({'success': False, 'error': 'Hệ thống AI bận: Dịch vụ AI tạm thời không khả dụng.'}), 500

# Chỉ dùng khi chạy file này độc lập dưới máy tính cá nhân
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
