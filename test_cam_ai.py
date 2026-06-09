import cv2
from ultralytics import YOLO

def main():
    print("⏳ Đang khởi động AI phân loại rác MindGuard...")
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ LỖI: Không mở được Camera.")
        return

    print("✅ Hệ thống SẴN SÀNG! Hãy thử giơ chai nhựa, điện thoại hoặc quả trái cây lên nhé.")

    while True:
        success, frame = cap.read()
        if not success: break

        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = int(box.conf[0] * 100)
                original_name = model.names[cls] # Tên tiếng Anh gốc của AI

                if conf > 40: # Độ tự tin trên 40%
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # ---------------- ZÔ PHẦN LOGIC PHÂN LOẠI RÁC ----------------
                    label = f"{original_name}" # Mặc định nếu không thuộc nhóm rác
                    color = (255, 255, 255)    # Mặc định chữ màu trắng

                    # 1. Nhóm Rác Tái Chế (Chai, ly nhựa, thủy tinh, tập sách)
                    if original_name in ['bottle', 'cup', 'wine glass', 'can', 'book']:
                        label = f"RAC TAI CHE ({original_name})"
                        color = (0, 255, 0) # Màu Xanh Lá

                    # 2. Nhóm Rác Hữu Cơ / Thực phẩm (Táo, chuối, cam, bánh mì...)
                    elif original_name in ['apple', 'banana', 'orange', 'sandwich', 'broccoli', 'cake']:
                        label = f"RAC HUU CO ({original_name})"
                        color = (0, 165, 255) # Màu Cam hoàng hôn

                    # 3. Nhóm Rác Nguy Hiểm / Điện Tử (Điện thoại, máy tính, remote, kéo)
                    elif original_name in ['cell phone', 'laptop', 'mouse', 'keyboard', 'remote', 'scissors']:
                        label = f"RAC DIEN TU ({original_name})"
                        color = (0, 0, 255) # Màu Đỏ cảnh báo

                    # ---------------- VẼ KHUNG VÀ CHỮ LÊN MÀN HÌNH ----------------
                    # Vẽ khung hình chữ nhật quanh vật thể
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Tạo nền đen nhỏ phía trên khung để chữ hiển thị rõ hơn
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label)*11, y1), color, -1)
                    
                    # Viết chữ phân loại rác (chữ màu trắng nền màu theo loại rác)
                    cv2.putText(frame, f"{label} {conf}%", (x1 + 5, y1 - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("MINDGUARD - AI TRASH SCANNER V2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()