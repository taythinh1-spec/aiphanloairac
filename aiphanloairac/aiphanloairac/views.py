import json
import logging
import os
from datetime import datetime

# === THƯ VIỆN BÊN NGOÀI ===
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

# ============================================================
# KHỞI TẠO
# ============================================================

# load_dotenv() đọc file .env khi chạy Local.
# Trên Render, biến môi trường đã được inject sẵn — hàm này bỏ qua an toàn.
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================
# KẾT NỐI MONGODB ATLAS (Connection Pool — khởi tạo một lần)
# ============================================================

collection = None  # Mặc định None; sẽ gán nếu kết nối thành công

_mongo_uri = os.environ.get("MONGO_URI")
if _mongo_uri:
    try:
        _client = MongoClient(_mongo_uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")  # Xác nhận kết nối thực sự
        _db = _client["AIPhanLoaiRac"]
        collection = _db["LichSuQuet"]
        logger.info("Đã kết nối MongoDB Atlas thành công.")
    except ConnectionFailure as e:
        logger.error("Lỗi kết nối MongoDB: %s", e)
else:
    logger.warning("Chưa cấu hình MONGO_URI — lịch sử quét sẽ không được lưu.")

# ============================================================
# HẰNG SỐ
# ============================================================

GEMINI_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-1.5-flash:generateContent"
)

GEMINI_PROMPT = (
    "Bạn là một chuyên gia phân loại rác thông minh. "
    "Hãy nhìn vào bức ảnh này và thực hiện nhiệm vụ sau:\n"
    "1. Xác định tên vật thể/loại rác xuất hiện trong ảnh.\n"
    "2. Phân loại nó vào một trong ba nhóm chính xác: "
    "'Rác hữu cơ', 'Rác vô cơ' hoặc 'Rác tái chế'.\n"
    "3. Đưa ra hướng dẫn xử lý hoặc vứt bỏ ngắn gọn, "
    "thân thiện với môi trường.\n"
    "Câu trả lời phải viết hoàn toàn bằng tiếng Việt, "
    "ngắn gọn, súc tích và dễ hiểu."
)

# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def extract_base64(image_data: str) -> str:
    """Tách phần dữ liệu thuần Base64, bỏ header 'data:image/...;base64,'."""
    if "," in image_data:
        return image_data.split(",", 1)[1]
    return image_data


def call_gemini(api_key: str, image_b64: str) -> str:
    """
    Gọi Gemini 1.5 Flash qua REST API với ảnh Base64.
    Trả về chuỗi kết quả phân loại.
    Ném RuntimeError nếu API trả về lỗi.
    """
    payload = {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64,
                    }
                },
            ]
        }]
    }

    resp = requests.post(
        GEMINI_URL,
        headers={"Content-Type": "application/json"},
        params={"key": api_key},   # Key nằm ở query param, không lộ trong body log
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200 or "candidates" not in resp.json():
        logger.error("Gemini API lỗi %s: %s", resp.status_code, resp.text[:300])
        raise RuntimeError("Gemini API không phản hồi hợp lệ.")

    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def save_history(image_b64: str, result: str) -> None:
    """Lưu kết quả quét vào MongoDB. Bỏ qua im lặng nếu chưa kết nối."""
    if collection is None:
        return
    try:
        collection.insert_one({
            "ket_qua": result,
            "thoi_gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hinh_anh": image_b64,
        })
        logger.info("Đã lưu lịch sử quét vào MongoDB.")
    except PyMongoError as e:
        # Lỗi lưu DB không nên làm hỏng response trả về user
        logger.error("Không thể lưu lịch sử: %s", e)

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    """Trang chủ giao diện chính."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Nhận ảnh Base64 từ camera → gọi Gemini AI phân loại
    → lưu lịch sử MongoDB → trả JSON về frontend.
    """
    try:
        # --- Bước 1: Kiểm tra đầu vào ---
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({
                "success": False,
                "error": "Không nhận được dữ liệu ảnh từ Camera.",
            }), 400

        # --- Bước 2: Kiểm tra API Key ---
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("Chưa cấu hình GEMINI_API_KEY trong Environment Variables!")
            return jsonify({
                "success": False,
                "error": "Hệ thống chưa cấu hình API Key.",
            }), 500

        # --- Bước 3: Gọi Gemini AI ---
        image_b64 = extract_base64(data["image"])
        try:
            ai_output = call_gemini(api_key, image_b64)
        except (requests.RequestException, RuntimeError) as e:
            logger.error("Lỗi gọi Gemini: %s", e)
            return jsonify({
                "success": False,
                "error": "Dịch vụ AI tạm thời không khả dụng, vui lòng thử lại.",
            }), 502

        logger.info("AI phân tích ảnh thành công.")

        # --- Bước 4: Lưu lịch sử (không chặn response nếu lỗi) ---
        save_history(data["image"], ai_output)

        # --- Bước 5: Trả kết quả ---
        return jsonify({"success": True, "prediction": ai_output})

    except Exception:
        logger.exception("Lỗi không xác định tại /predict")
        return jsonify({
            "success": False,
            "error": "Đã xảy ra lỗi máy chủ.",
        }), 500


# ============================================================
# ĐIỂM KHỞI ĐỘNG
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=False)
