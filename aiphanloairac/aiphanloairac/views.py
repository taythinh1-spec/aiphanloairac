import os
import base64
import re
import json
import io
from flask import Flask, render_template, request, jsonify
from google import genai
from PIL import Image

app = Flask(__name__, template_folder='templates')

# Khởi tạo Client Gemini thế hệ mới
client = genai.Client()

# RAM database tạm thời lưu điểm
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
        
        username = data.get('username', 'Học sinh ẩn danh').strip() or 'Học sinh ẩn danh'

        # 1. Giải mã dữ liệu ảnh Base64
        image_data = data['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))

        # 2. Prompt ép Gemini quét nhiều món rác và trả về cấu trúc JSON
        prompt = """
        Bạn là một chuyên gia AI về môi trường học đường.
        Hãy phân tích bức ảnh và phát hiện tất cả các món rác xuất hiện trong hình.
        Trả về một chuỗi JSON thuần túy (tuyệt đối không để trong ký hiệu markdown ```json), cấu trúc như sau:
        {
          "danh_sach_rac": [
            {
              "ten": "Tên đồ vật bằng tiếng Việt",
              "nhom": "Rác Tái Chế hoặc Rác Nguy Hại hoặc Rác Hữu Cơ hoặc Rác Còn Lại",
              "huong_dan": "Cách phân loại ngắn gọn",
              "diem": số_điểm_nguyên
            }
          ],
          "tac_hai": "1 câu ngắn về tác hại môi trường của các loại rác này",
          "meo": "1 mẹo nhỏ cho học sinh xử lý nhanh rác này trước khi vứt"
        }
        """

        # 3. Gọi API Gemini 2.5 Flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image]
        )

        # 4. Parse dữ liệu và ép kiểu JSON
        text_data = response.text.strip().replace("```json", "").replace("```", "").strip()
        result_json = json.loads(text_data)
        
        danh_sach = result_json.get("danh_sach_rac", [])
        tac_hai = result_json.get("tac_hai", "Chưa xác định tác hại.")
        meo = result_json.get("meo", "Hãy vứt rác đúng nơi quy định.")

        # 5. XỬ LÝ NỐI CHUỖI ĐỂ TẠO RA KẾT QUẢ DÀI VÀ CHI TIẾT
        # Đoạn này sẽ biến đổi danh sách thành 1 khối văn bản hiển thị cực xịn trên HTML cũ
        chuoi_hien_thi = "🔍 [KẾT QUẢ QUÉT ĐA VẬT THỂ]\n"
        tong_diem_anh_nay = 0
        
        for i, item in enumerate(danh_sach, 1):
            ten_mon = item.get("ten", "Không rõ").upper()
            nhom_rac = item.get("nhom", "Rác Còn Lại")
            hd_xu_ly = item.get("huong_dan", "Bỏ vào thùng rác")
            diem_mon = int(item.get("diem", 2))
            
            tong_diem_anh_nay += diem_mon
            
            # Tạo dòng thông tin cho từng món rác
            chuoi_hien_thi += f"\n📦 VẬT THỂ #{i}: {ten_mon} ({nhom_rac})"
            chuoi_hien_thi += f"\n👉 Hướng dẫn: {hd_xu_ly}"
            chuoi_hien_thi += f"\n⭐ Điểm: +{diem_mon}đ\n"
            
        chuoi_hien_thi += f"\n⚠️ TÁC HẠI: {tac_hai}"
        chuoi_hien_thi += f"\n💡 MẸO HAY: {meo}"

        # 6. Tích lũy điểm hệ thống
        if username not in USER_POINTS_DB:
            USER_POINTS_DB[username] = 0
        USER_POINTS_DB[username] += tong_diem_anh_nay

        # TRẢ VỀ: Giữ nguyên các key 'prediction', 'diem_cong_tu_ai', 'tong_diem_he_thong' để khớp với HTML cũ
        return jsonify({
            'success': True,
            'prediction': chuoi_hien_thi, 
            'diem_cong_tu_ai': tong_diem_anh_nay,
            'tong_diem_he_thong': USER_POINTS_DB[username]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f"Lỗi hệ thống: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
