import os
import io
import base64
import re
import json
from flask import Flask, render_template, request, jsonify
from PIL import Image
import google.generativeai as genai

# Khởi tạo Flask App
app = Flask(__name__, template_folder='templates')

# 1. Cấu hình Gemini API
# Bạn nên lấy API Key từ Environment Variable của Render để bảo mật
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Sử dụng mô hình Gemini 1.5 Flash siêu nhanh và tiết kiệm
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

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

        # 2. Xử lý ảnh Base64 và chuyển thành định dạng mà thư viện Pillow (PIL) hiểu được
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = base64.b64decode(image_data)
        
        try:
            image = Image.open(io.BytesIO(img_bytes))
        except Exception:
            return jsonify({'success': False, 'error': 'Dữ liệu hình ảnh bị hỏng hoặc không đúng định dạng!'}), 400

        # 3. Tạo prompt thiết lập "vị trí" và "quy tắc trả về dữ liệu" cho Gemini
        prompt = """
        Bạn là một chuyên gia AI phân loại rác thân thiện tại trường học. 
        Hãy phân tích hình ảnh vật thể/rác này và trả về kết quả chính xác dưới duy nhất một định dạng JSON (vui lòng KHÔNG bọc trong ký tự markdown như ```json ... ```, chỉ trả về chuỗi text JSON thuần túy).
        
        Cấu trúc JSON bắt buộc phải như sau:
        {
          "loai_rac": "Tên nhóm rác ngắn gọn (Ví dụ: Chai lọ nhựa / Lon nhôm tái chế, Sách báo / Giấy vụn, Thức ăn thừa / Vỏ trái cây, Thiết bị điện tử / Vật sắc nhọn, hoặc Vật thể lạ / Rác vô cơ khác)",
          "huong_dan": "Lời khuyên ngắn gọn, dễ thương hướng dẫn học sinh bỏ vào đúng thùng màu gì ở trường và cách xử lý trước khi bỏ rác",
          "diem_cong": <Số điểm cộng dạng số nguyên: Tái chế/Giấy được 10 điểm, Hữu cơ được 5 điểm, Nguy hiểm/Điện tử được 20 điểm, Rác vô cơ khác được 2 điểm>
        }
        """

        # 4. Gửi ảnh và prompt sang Google Gemini để xử lý
        response = model_gemini.generate_content([prompt, image])
        
        # Làm sạch chuỗi phản hồi phòng trường hợp AI tự thêm ký tự lạ
        response_text = response.text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```[a-zA-Z]*\n|```$', '', response_text, flags=re.MULTILINE).strip()

        # 5. Ép kiểu chuỗi nhận được thành JSON của Python
        ai_result = json.loads(response_text)
        
        loai_rac = ai_result.get("loai_rac", "Vật thể lạ / Rác vô cơ khác")
        huong_dan = ai_result.get("huong_dan", "Hãy vứt vào thùng rác quy định nhé!")
        diem_cong = int(ai_result.get("diem_cong", 2))

        # Định dạng chuỗi hiển thị đúng như giao diện cũ của bạn yêu cầu
        chuoi_hien_thi = f"📍 ĐỒ VẬT: {loai_rac.upper()}\n💡 GIẢI PHÁP: {huong_dan}"

        # 6. Lưu dữ liệu điểm thi đua
        if username not in USER_POINTS_DB:
            USER_POINTS_DB[username] = 0
        USER_POINTS_DB[username] += diem_cong

        return jsonify({
            'success': True,
            'prediction': chuoi_hien_thi,
            'diem_cong_tu_ai': diem_cong,
            'tong_diem_he_thong': USER_POINTS_DB[username]
        })

    except json.JSONDecodeError:
        return jsonify({'success': False, 'error': 'Lỗi xử lý cấu trúc dữ liệu từ AI!'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
