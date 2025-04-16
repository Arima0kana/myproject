import socket
import cv2
import numpy as np

# 配置参数
HOST = '0.0.0.0'  # 监听所有IP
PORT = 5000        # 端口

# 创建socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print("等待连接")

conn, addr = s.accept()
print(f"已连接: {addr}")

# 窗口
cv2.namedWindow('Raspberry Pi Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Raspberry Pi Stream', 1080, 780)

while True:
    # 接收图像
    header = conn.recv(4)
    if not header: break
    
    img_size = int.from_bytes(header, byteorder='big')
    
    # 接收图像数据
    chunks = []
    bytes_received = 0
    while bytes_received < img_size:
        chunk = conn.recv(min(img_size - bytes_received, 4096))
        if not chunk: break
        chunks.append(chunk)
        bytes_received += len(chunk)
    
    # 解码
    img_data = b''.join(chunks)
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    
    if img is not None:
        cv2.imshow('Raspberry Pi Stream', img)
        if cv2.waitKey(1) == 27:  # ESC键退出
            break

# 释放
conn.close()
cv2.destroyAllWindows()