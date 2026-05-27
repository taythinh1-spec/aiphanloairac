import logging
import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

# ============================================================
# KHỞI TẠO
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================
# KẾT NỐI MONGODB ATLAS
# ============================================================

collection = None

_mongo_uri = os.environ.get("MONGO_URI")
if _mongo_uri:
    try:
        _client = MongoClient(_mongo_uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
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

# Đã xóa GEMINI_URL ở đây để hệ thống tự động quét đường dẫn

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

def extract_base64(image_data: str) -> tuple[str, str]:
    """
    Tách Base64 thuần và tự động nhận diện mime_type từ header.
    Trả về (base64_string, mime_type).
    Mặc định mime_type là image/jpeg nếu không đọc được header.
    """
    mime_type = "image/jpeg"  # fallback an toàn
    if "," in image_data:
        header, data = image_data.split(",", 1)
        # header có dạng: data:image/png;base64
        if "image/png" in header:
            mime_type = "image/png"
        elif "image/webp" in header:
            mime_type = "image/webp"
        elif "image/jpeg" in header or "image/jpg" in header:
            mime_type = "image/jpeg"
        return data, mime_type
    return image_data, mime_type


def get_best_model(api_key: str) -> str:
    """
    Tự động gọi API của Google để lấy danh sách các model khả dụng cho tài khoản.
    Tự động lọc và trả về tên model dòng 'flash' hỗ trợ 'generateContent'.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            available_models = data.get("models", [])
            
            # Ưu tiên 1: Tìm model có chữ 'flash' và hỗ trợ 'generateContent'
            for model in available_models:
                name = model.get("name", "")
                methods = model.get("supportedGenerationMethods", [])
                if "generateContent" in methods and "flash" in name:
                    logger.info("Đã tự động tìm thấy model tối ưu: %s", name)
                    return name
            
            # Ưu tiên 2: Nếu không có flash, lấy model bất kỳ hỗ trợ generateContent
            for model in available_models:
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    name = model.get("name")
                    logger.warning("Không tìm thấy Flash, dùng model thay thế: %s", name)
                    return name
                    
    except Exception as e:
        logger.error("Lỗi khi tự động quét model: %s", e)
    
    # Mặc định an toàn nếu quá trình quét bị lỗi
    return "models/gemini-1.5-flash"


def call_gemini(api_key: str, image_b64: str, mime_type: str) -> str:
    """
    Gọi Gemini qua REST API. Tự động lấy cấu hình model khả dụng nhất.
    Nhận mime_type động để xử lý đúng PNG/JPEG/WebP từ camera.
    Ném RuntimeError nếu API trả về lỗi.
    """
    # --- TỰ ĐỘNG LẤY TÊN MODEL ---
    model_name = get_best_model(api_key)
    
    # --- TỰ ĐỘNG TẠO ĐƯỜNG DẪN KẾT NỐI ---
    dynamic_url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"

    payload = {
        "contents": [{
            "parts": [
                {"text": GEMINI_PROMPT},
                {
                    "inline_data": {
                        "mime_type": mime_type,  # ← động, không hardcode
                        "data": image_b64,
                    }
                },
            ]
        }]
    }

    resp = requests.post(
        dynamic_url,
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        json=payload,
        timeout=30,
    )

    resp_json = resp.json()

    # Log chi tiết để dễ debug nếu còn lỗi
    if resp.status_code != 200 or "candidates" not in resp_json:
        logger.error(
            "Gemini API lỗi — status: %s | URL: %s | mime: %s | response: %s",
            resp.status_code,
            dynamic_url,
            mime_type,
            resp.text[:500],
        )
        raise RuntimeError("Gemini API không phản hồi hợp lệ.")

    return resp_json["candidates"][0]["content"]["parts"][0]["text"]


def save_history(image_data: str, result: str) -> None:
    """Lưu kết quả quét vào MongoDB. Bỏ qua im lặng nếu chưa kết nối."""
    if collection is None:
        return
    try:
        collection.insert_one({
            "ket_qua": result,
            "thoi_gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hinh_anh": image_data,
        })
        logger.info("Đã lưu lịch sử quét vào MongoDB.")
    except PyMongoError as e:
        logger.error("Không thể lưu lịch sử: %s", e)

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
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

        # --- Bước 3: Tách Base64 và nhận diện định dạng ảnh ---
        image_b64, mime_type = extract_base64(data["image"])
        logger.info("Nhận ảnh định dạng: %s", mime_type)

        # --- Bước 4: Gọi Gemini AI ---
        try:
            ai_output = call_gemini(api_key, image_b64, mime_type)
        except (requests.RequestException, RuntimeError) as e:
            logger.error("Lỗi gọi Gemini: %s", e)
            return jsonify({
                "success": False,
                "error": "Dịch vụ AI tạm thời không khả dụng, vui lòng thử lại.",
            }), 502

        logger.info("AI phân tích ảnh thành công.")

        # --- Bước 5: Lưu lịch sử ---
        save_history(data["image"], ai_output)

        # --- Bước 6: Trả kết quả ---
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
