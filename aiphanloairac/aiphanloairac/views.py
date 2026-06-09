import base64
import io
import json
import logging
import os

# === THƯ VIỆN BÊN NGOÀI ===
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

# ============================================================
# CẤU HÌNH KHỞI TẠO
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ============================================================
# KHỞI TẠO CÁC DỊCH VỤ (Chạy một lần khi server khởi động)
# ============================================================

def _init_gemini():
    """Khởi tạo model Google Gemini AI từ biến môi trường."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Thiếu biến môi trường API Key của Gemini trong cấu hình")
    genai.configure(api_key=api_key)
    logger.info("Đã cấu hình Google Gemini AI thành công.")
    
    # Đã sửa đổi: Thêm tiền tố 'models/' để định danh chính xác trên API của Render
    return genai.GenerativeModel("models/gemini-1.5-flash")


def _init_mongodb():
    """Khởi tạo kết nối MongoDB Atlas với Connection Pooling."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Thiếu biến môi trường MONGO_URI")

    # Tăng thời gian chờ lên 10s đề phòng mạng Render kết nối sang Atlas bị chậm lúc khởi động
    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    
    # Kiểm tra kết nối bằng lệnh ping
    mongo_client.admin.command("ping")
    logger.info("Đã kết nối MongoDB Atlas thành công.")
    return mongo_client


# Khởi động dịch vụ hệ thống an toàn
try:
    gemini_model = _init_gemini()
    db_client = _init_mongodb()
    users_collection = db_client["CuocThiSangTao"]["XepHangHocSinh"]
except (ValueError, ConnectionFailure, Exception) as e:
    logger.critical("Lỗi khởi tạo dịch vụ hệ thống: %s", e)
    # Không làm sập ngang server Render, gán None để handle khéo léo ở route
    gemini_model = None
    db_client = None
    logger.warning("Hệ thống khởi động ở chế độ dự phòng (Mất kết nối dịch vụ).")


# ============================================================
# CẤU HÌNH PHÂN LOẠI RÁC
# ============================================================

# Bảng tra cứu điểm và nhãn hiển thị map chuẩn cấu hình thi đua học đường
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
# HÀM TIỆN ÍCH TỐI ƯU
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


def update_user_score(username: str, diem_cong: int) -> int:
    """Cộng điểm tích lũy vào MongoDB (Upsert) và lấy tổng điểm mới nhất."""
    if users_collection is None:
        raise PyMongoError("Kết nối cơ sở dữ liệu chưa sẵn sàng.")
        
    users_collection.update_one(
        {"username": username},
        {"$inc": {"diem": diem_cong}},
        upsert=True,
    )
    user_info = users_collection.find_one({"username": username})
    return user_info["diem"] if user_info else diem_cong


# ============================================================
# CÁC ROUTE (ĐƯỜNG DẪN WEB)
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Xử lý phân loại rác bằng Gemini JSON Mode và lưu điểm thi đua học sinh."""
    # Kiểm tra xem dịch vụ AI có khởi động lỗi từ trước không
    if gemini_model is None:
        return jsonify({"success": False, "error": "Hệ thống AI chưa được cấu hình hoặc lỗi API Key."}), 503

    try:
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        username = (data.get("username", "") or "Hoc_Sinh_An_Danh").strip()

        if not image_data:
            return jsonify({"success": False, "error": "Không nhận được dữ liệu ảnh từ camera."}), 400

        # --- Bước 1: Giải mã hình ảnh từ Client ---
        try:
            image = decode_base64_image(image_data)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400

        # --- Bước 2: Gọi Gemini AI phản hồi dạng JSON chuẩn ---
        try:
            response = gemini_model.generate_content(
                [GEMINI_PROMPT, image],
                generation_config={"response_mime_type": "application/json"}
            )
            # Chuyển đổi dữ liệu chuỗi JSON từ AI thành Dictionary Python
            ai_data = json.loads(response.text.strip())
        except Exception as e:
            logger.error("Lỗi tương tác Gemini API hoặc cấu trúc trả về: %s", e)
            return jsonify({"success": False, "error": "Dịch vụ AI tạm thời gặp sự cố kết nối."}), 503

        # Trích xuất dữ liệu an toàn từ kết quả của AI
        loai_ai = ai_data.get("loai", "Vo_Co").strip()
        ten_mon_do = ai_data.get("ten", "Vật thể lạ").strip()
        loi_khuyen = ai_data.get("loi_khuyen", "Hãy vứt vào thùng rác quy định.").strip()

        # --- Bước 3: Tra cứu hệ thống điểm thi đua ---
        if loai_ai not in WASTE_CATEGORIES:
            loai_ai = "Vo_Co"  # Hạ cấp an toàn nếu AI trả về phân loại nằm ngoài danh mục
        diem_cong, nhan_loai = WASTE_CATEGORIES[loai_ai]

        # Định dạng cấu trúc hiển thị đồng bộ tuyệt đối với Frontend cũ
        chuoi_hien_thi = f"📍 ĐỒ VẬT: {ten_mon_do.upper()} -> {nhan_loai.upper()}\n💡 GIẢI PHÁP: {loi_khuyen}"

        # --- Bước 4: Cập nhật bảng xếp hạng thi đua học sinh ---
        try:
            tong_diem = update_user_score(username, diem_cong)
        except (PyMongoError, Exception) as e:
            logger.error("Lỗi đồng bộ dữ liệu điểm MongoDB: %s", e)
            return jsonify({"success": False, "error": "Lỗi hệ thống lưu trữ điểm số học sinh."}), 503

        # --- Bước 5: Trả dữ liệu đúng Key Frontend đang mong đợi ---
        return jsonify({
            "success": True,
            "prediction": chuoi_hien_thi,         # Đúng Key frontend nhận diện
            "diem_cong_tu_ai": diem_cong,         # Đúng Key frontend nhận diện
            "tong_diem_he_thong": tong_diem,      # Đúng Key frontend nhận diện
        })

    except Exception as e:
        logger.exception("Lỗi không lường trước tại /predict")
        return jsonify({"success": False, "error": "Đã xảy ra lỗi hệ thống máy chủ."}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
