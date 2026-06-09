import base64
import io
import json
import logging
import os

# === THƯ VIỆN CHUẨN THẾ HỆ MỚI ===
from google import genai
from google.genai import types
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image
from pymongo import MongoClient

# ============================================================
# CẤU HÌNH KHỞI TẠO FLASK
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Bảng tra cứu điểm và nhãn hiển thị theo chuẩn thi đua học đường
WASTE_CATEGORIES = {
    "Tai_Che": (10, "Chai lọ nhựa / Lon nhôm tái chế ♻️"),
    "Huu_Co":  (5,  "Thức ăn thừa / Vỏ trái cây 🌱"),
    "Vo_Co":   (2,  "Vật thể lạ / Rác vô cơ khác 🗑️"),
}

# Prompt yêu cầu Gemini ép cấu trúc dữ liệu JSON thuần túy
GEMINI_PROMPT = (
    "Bạn là một chuyên gia phân loại rác thải bảo vệ môi trường tại trường học.\n"
    "Hãy phân tích bức ảnh này và trả về kết quả phân loại dưới dạng cấu trúc JSON sau:\n"
    "{\n"
    "  \"loai\": \"Điền một trong ba từ chính xác: Tai_Che hoặc Huu_Co hoặc Vo_Co\",\n"
    "  \"ten\": \"Tên món đồ bằng tiếng Việt ngắn gọn\",\n"
    "  \"loi_khuyen\": \"Lời khuyên dễ thương hướng dẫn học sinh cách xử lý bỏ vào thùng rác nào\"\n"
    "}"
)

# ============================================================
# HÀM TIỆN ÍCH GIẢI MÃ ẢNH
# ============================================================

def decode_base64_image(image_data: str) -> Image.Image:
    """Giải mã chuỗi Base64 từ frontend thành đối tượng PIL Image."""
    try:
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
        image_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Dữ liệu ảnh không hợp lệ: {e}") from e

# ============================================================
# CÁC ROUTE XỬ LÝ (ĐƯỜNG DẪN WEB)
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Xử lý phân loại rác bằng Gemini JSON Mode thế hệ mới và lưu điểm vào MongoDB."""
    try:
        # 1. KIỂM TRA BIẾN MÔI TRƯỜNG KHI CÓ REQUEST GỬI LÊN
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        mongo_uri = os.getenv("MONGO_URI")
        
        if not api_key:
            return jsonify({"success": False, "error": "Hệ thống chưa cấu hình GEMINI_API_KEY trên Render."}), 503
        if not mongo_uri:
            return jsonify({"success": False, "error": "Hệ thống chưa cấu hình MONGO_URI trên Render."}), 503

        # 2. ĐỌC DỮ LIỆU TỪ FRONTEND ĐƯỢC CAMERA GỬI LÊN
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        username = (data.get("username", "") or "Hoc_Sinh_An_Danh").strip()

        if not image_data:
            return jsonify({"success": False, "error": "Không nhận được dữ liệu ảnh từ camera."}), 400

        # --- Bước 1: Giải mã hình ảnh ---
        try:
            image = decode_base64_image(image_data)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400

        # --- Bước 2: Gọi Gemini API bằng Client Thế Hệ Mới ---
        try:
            # Khởi tạo client cục bộ bảo mật, tự động tối ưu luồng gọi trên Cloud
            client_ai = genai.Client(api_key=api_key)
            
            response = client_ai.models.generate_content(
                model='gemini-1.5-flash',
                contents=[image, GEMINI_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            ai_data = json.loads(response.text.strip())
        except Exception as e:
            logger.error("Lỗi tương tác Gemini API thế hệ mới: %s", e)
            return jsonify({"success": False, "error": "Dịch vụ AI đang bận, vui lòng thử lại sau vài giây."}), 503

        # Trích xuất dữ liệu an toàn từ kết quả của AI
        loai_ai = ai_data.get("loai", "Vo_Co").strip()
        ten_mon_do = ai_data.get("ten", "Vật thể lạ").strip()
        loi_khuyen = ai_data.get("loi_khuyen", "Hãy vứt vào thùng rác quy định.").strip()

        # --- Bước 3: Tra cứu hệ thống điểm thi đua ---
        if loai_ai not in WASTE_CATEGORIES:
            loai_ai = "Vo_Co"
        diem_cong, nhan_loai = WASTE_CATEGORIES[loai_ai]

        chuoi_hien_thi = f"📍 ĐỒ VẬT: {ten_mon_do.upper()} -> {nhan_loai.upper()}\n💡 GIẢI PHÁP: {loi_khuyen}"

        # --- Bước 4: Khởi tạo kết nối MongoDB động để cộng điểm thi đua ---
        try:
            with MongoClient(mongo_uri, serverSelectionTimeoutMS=5000) as client_db:
                users_collection = client_db["CuocThiSangTao"]["XepHangHocSinh"]
                users_collection.update_one(
                    {"username": username},
                    {"$inc": {"diem": diem_cong}},
                    upsert=True,
                )
                user_info = users_collection.find_one({"username": username})
                tong_diem = user_info["diem"] if user_info else diem_cong
        except Exception as e:
            logger.error("Lỗi đồng bộ dữ liệu điểm MongoDB: %s", e)
            return jsonify({"success": False, "error": "Lỗi kết nối cơ sở dữ liệu điểm số học sinh."}), 503

        # --- Bước 5: Trả dữ liệu map chuẩn Frontend ---
        return jsonify({
            "success": True,
            "prediction": chuoi_hien_thi,
            "diem_cong_tu_ai": diem_cong,
            "tong_diem_he_thong": tong_diem,
        })

    except Exception as e:
        logger.exception("Lỗi hệ thống nghiêm trọng tại /predict")
        return jsonify({"success": False, "error": "Đã xảy ra lỗi máy chủ không mong muốn."}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
