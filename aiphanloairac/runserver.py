import os
import sys

# Thêm thư mục hiện tại vào đường dẫn tìm kiếm của Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from views import app

if __name__ == '__main__':
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)