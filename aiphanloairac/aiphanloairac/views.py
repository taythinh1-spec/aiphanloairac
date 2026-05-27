import base64
import io
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

# Tải biến môi trường từ file .env (không bao giờ hardcode API key)
load_dotenv()

# Thiết lập logging để dễ theo dõi lỗi trên Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ============================================================
# KHỞI TẠO CÁC DỊCH VỤ (chỉ chạy một lần khi server khởi động)
# ============================================================

def _init_gemini():
    """Khởi tạo model Google Gemini AI từ biến môi trường."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Thiếu biến môi trường GOOGLE_API_KEY trong file .env")
    genai.configure(api_key=api_key)
    logger.info("Đã kết nối Google Gemini AI thành công.")
    return genai.GenerativeModel("gemini-1.5-flash")


def _init_mongodb():
    """
    Khởi tạo kết nối MongoDB Atlas với Connection Pooling.
    MongoClient tự động quản lý pool kết nối, không cần tạo lại mỗi request.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("Thiếu biến môi trường MONGO_URI trong file .env")

    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    # Kiểm tra kết nối thực sự ngay lúc khởi động
    mongo_client.admin.command("ping")
    logger.info("Đã kết nối MongoDB Atlas thành công.")
    return mongo_client


# Khởi tạo dịch vụ tại module level (Connection Pooling đúng chuẩn)
try:
    gemini_model = _init_gemini()
    db_client = _init_mongodb()
    users_collection = db_client["CuocThiSangTao"]["XepHangHocSinh"]
except (ValueError, ConnectionFailure) as e:
    logger.critical("Lỗi khởi tạo dịch vụ: %s", e)
    raise SystemExit(1) from e


# ============================================================
# CẤU HÌNH PHÂN LOẠI RÁC
# ============================================================

# Bảng tra cứu loại rác: từ khóa AI -> (điểm, nhãn hiển thị)
WASTE_CATEGORIES = {
    "Tai_Che": (10, "Rác Tái Chế (Giấy, chai lọ, kim loại) ♻️"),
    "Huu_Co":  (5,  "Rác Hữu Cơ (Thức ăn thừa, lá cây, vỏ quả) 🌱"),
    "Vo_Co":   (2,  "Rác Vô Cơ / Nguy Hại (Túi nilon, pin, hộp xốp) 🗑️"),
}

# Prompt gửi cho Gemini — yêu cầu trả về đúng cấu trúc để dễ parse
GEMINI_PROMPT = (
    "Bạn là một chuyên gia phân loại rác thải bảo vệ môi trường.\n"
    "Hãy nhìn vào bức ảnh này và phân loại chính xác món đồ theo đúng cấu trúc sau:\n"
    "Loại: [Điền một trong ba từ: Tai_Che hoặc Huu_Co hoặc Vo_Co]\n"
    "Tên: [Tên món đồ bằng tiếng Việt]\n"
    "Lời khuyên: [Cách xử lý hoặc một ý tưởng sáng tạo độc đáo để tái chế món đồ này]"
)


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def decode_base64_image(image_data: str) -> Image.Image:
    """
    Giải mã chuỗi Base64 từ frontend thành đối tượng PIL Image.
    Ném ValueError nếu dữ liệu không hợp lệ.
    """
    try:
        _, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Dữ liệu ảnh không hợp lệ: {e}") from e


def parse_gemini_response(response_text: str) -> dict:
    """
    Phân tích văn bản phản hồi từ Gemini thành từ điển có cấu trúc.
    Trả về giá trị mặc định nếu không tìm thấy trường nào.
    """
    result = {
        "loai": "Vo_Co",
        "ten": "Không xác định",
        "loi_khuyen": "Hãy bỏ vào thùng rác quy định.",
    }
    for line in response_text.strip().splitlines():
        if "Loại:" in line:
            result["loai"] = line.replace("Loại:", "").strip()
        elif "Tên:" in line:
            result["ten"] = line.replace("Tên:", "").strip()
        elif "Lời khuyên:" in line:
            result["loi_khuyen"] = line.replace("Lời khuyên:", "").strip()
    return result


def get_waste_info(loai: str) -> tuple[int, str]:
    """
    Tra cứu điểm số và nhãn hiển thị dựa theo từ khóa loại rác từ AI.
    Mặc định trả về loại Vo_Co nếu không khớp từ khóa nào.
    """
    for key, (diem, nhan) in WASTE_CATEGORIES.items():
        if key in loai:
            return diem, nhan
    return WASTE_CATEGORIES["Vo_Co"]


def update_user_score(username: str, diem_cong: int) -> int:
    """
    Cộng điểm cho học sinh vào MongoDB và trả về tổng điểm mới nhất.
    Tự động tạo document mới nếu học sinh chưa tồn tại (upsert=True).
    Ném PyMongoError nếu thao tác database thất bại.
    """
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
    """Trả về trang chủ của ứng dụng."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Nhận ảnh rác từ camera, gọi Gemini AI phân loại,
    cộng điểm vào MongoDB và trả về kết quả cho frontend.
    """
    try:
        data = request.get_json(force=True)
        image_data = data.get("image", "")
        username = (data.get("username", "") or "Hoc_Sinh_An_Danh").strip()

        # --- Bước 1: Kiểm tra đầu vào ---
        if not image_data:
            return jsonify({"success": False, "error": "Không nhận được ảnh từ camera."}), 400

        # --- Bước 2: Giải mã ảnh Base64 ---
        try:
            image = decode_base64_image(image_data)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400

        # --- Bước 3: Gọi Gemini AI phân loại ---
        try:
            response = gemini_model.generate_content([GEMINI_PROMPT, image])
        except Exception as e:
            logger.error("Lỗi Gemini API: %s", e)
            return jsonify({"success": False, "error": "Dịch vụ AI tạm thời không khả dụng."}), 503

        # --- Bước 4: Phân tích kết quả và tra cứu điểm ---
        parsed = parse_gemini_response(response.text)
        diem_cong, nhan_loai = get_waste_info(parsed["loai"])

        # --- Bước 5: Lưu điểm vào database ---
        try:
            tong_diem = update_user_score(username, diem_cong)
        except PyMongoError as e:
            logger.error("Lỗi MongoDB: %s", e)
            return jsonify({"success": False, "error": "Lỗi cơ sở dữ liệu, vui lòng thử lại."}), 503

        # --- Bước 6: Trả kết quả về frontend ---
        return jsonify({
            "success": True,
            "loai_rac": f"{parsed['ten']} -> {nhan_loai}",
            "huong_dan": parsed["loi_khuyen"],
            "diem_cong": diem_cong,
            "tong_diem": tong_diem,
        })

    except Exception as e:
        # Bắt mọi lỗi không lường trước, không để lộ chi tiết nội bộ ra ngoài
        logger.exception("Lỗi không xác định tại /predict")
        return jsonify({"success": False, "error": "Đã xảy ra lỗi máy chủ."}), 500


# ============================================================
# ĐIỂM KHỞI ĐỘNG
# ============================================================

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)