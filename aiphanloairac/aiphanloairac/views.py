import os
import base64
import re
import json
import io
from flask import Flask, render_template, request, jsonify
from google import genai
from PIL import Image

# Khởi tạo Flask App
app = Flask(__name__, template_folder='templates')

# Khởi tạo Client thế hệ mới (Tự động nhận biến môi trường GEMINI_API_KEY từ Render)
# Cú pháp mới siêu gọn, không cần cấu hình configure rườm rà
client = genai.Client()

# RAM database tạm thời lưu điểm thi đua
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
            return jsonify({'success': False, 'error': 'Không nhận được dữ liệu ảnh từ Camera!'}), 400
        
        username = data.get('username', 'Học sinh ẩn danh').strip()
        if not username:
            username = 'Học sinh ẩn danh'

        # 1. Giải mã dữ liệu ảnh Base64 gửi từ trình duyệt
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = base64.b64decode(image_data)

        # Chuyển đổi bytes ảnh sang đối tượng PIL Image để tương thích tuyệt đối với SDK mới
        image = Image.open(io.BytesIO(img_bytes))

        # 2. Câu lệnh Prompt điều khiển AI phân loại rác học đường
        prompt = """
        Bạn là một trợ lý AI tích hợp trong thùng rác thông minh học đường. 
        Hãy phân tích bức ảnh được chụp từ camera này thật nhanh và trả về kết quả phân loại rác dưới dạng một chuỗi JSON thuần túy (tuyệt đối không để trong ký hiệu toán học hoặc markdown ```json), cấu trúc bắt buộc như sau:
        {
          "loai_rac": "Tên đồ vật bằng tiếng Việt (Ví dụ: Chai nhựa, Hộp sữa giấy, Thiết bị điện tử, Vỏ bánh nilon...)",
          "huong_dan": "Gợi ý hành động ngắn gọn cho học sinh (Ví dụ: Bỏ vào thùng rác tái chế màu xanh lá, mang tới khu gom pin cũ...)",
          "diem_cong": số_điểm_thưởng_là_số_nguyên (Quy ước: Rác tái chế giấy/nhựa cộng 10; Thiết bị nguy hiểm/Điện tử/Pin cũ cộng 20; Thức ăn hữu cơ cộng 5; Rác vô cơ khác cộng 2)
        }
        """

        # 3. Gọi API thế hệ mới bằng Model Flash tối tân, chạy qua cổng v1 chuẩn hóa
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image]
        )

        # 4. Xử lý và chuẩn hóa chuỗi dữ liệu nhận được từ Google
        text_data = response.text.strip()
        text_data = text_data.replace("```json", "").replace("```", "").strip()

        # Ép kiểu chuỗi text về Dictionary JSON
        result_json = json.loads(text_data)
        
        loai_rac = result_json.get("loai_rac", "Vật thể lạ")
        huong_dan = result_json.get("huong_dan", "Vui lòng phân loại vào thùng rác sinh hoạt thông thường.")
        diem_cong = int(result_json.get("diem_cong", 2))

        # Định dạng chuỗi hiển thị lên màn hình giao diện Cyberpunk
        chuoi_hien_thi = f"📍 ĐỒ VẬT: {loai_rac.upper()}\n💡 GIẢI PHÁP: {huong_dan}"

        # 5. Tích lũy quỹ điểm thi đua cho học sinh
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
        return jsonify({'success': False, 'error': f"Lỗi tương tác Gemini API thế hệ mới: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
