import cv2
import os
import sys
from ultralytics import YOLO
import numpy as np
import time
from datetime import datetime
from djitellopy import Tello
import threading
import logging

# cac bien can thiet de tao o phan trên cua chuong trinh
photos_taken = 0
photos_to_take = 10

# Lấy ngày và giờ hiện tại
current_datetime = datetime.now()

# Định dạng ngày tháng
formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

# Chuyển hướng stderr để không in các thông báo lỗi từ libav.h264
sys.stderr = open(os.devnull, 'w')
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('djitellopy')
logger.setLevel(logging.WARNING)

# Load the YOLOv8 model
model = YOLO('yolov9c.pt')

# dài rộng của khung hình
video_w = 1280
video_h = 800

# Biến đếm khung hình
frame_count = 0

# Tính điểm trung tâm của khung hình
center_x_frame = video_w / 2
center_y_frame = video_h / 2
dead_zone = 5000
ideal_areas = list(range(100))


pid_x = [0.2, 0.5, 0] #pid yaw
pid_y = [0.4, 2, 0] #pid up down
pid_z = [0.0008, 0.005, 0] #pid front back

x_error = 0
y_error = 0
z_error = 0
x_error_new = 0
y_error_new = 0
z_error_new = 0


#thêm biến để theo dõi thời gian cuối cùng mà lệnh được gửi
last_update_time = time.time()

# Setup time for FPS
times = []

# Drone DJI Tello
me = Tello()
me.connect()
me.streamon()
me.takeoff()
time.sleep(4)
me.move_up(120)
print(me.get_battery())
me.get_height()
yaw_velocity = 0
up_down_velocity = 0
for_back_velocity = 0
left_right_velocity = 0

id_tracking = [-1]
def get_id(id_list):
    while True:
        track_id = int(input("Please enter your tracking ID: "))
        id_list[0] = track_id
        if track_id == -2:
            break

threading.Thread(target=get_id, args=(id_tracking,)).start()

# Đường dẫn cơ bản và thư mục lưu trữ
base_folder = 'luuvaoday'
image_folder = os.path.join(base_folder, 'luuanh')
video_folder = os.path.join(base_folder, 'luuvideo')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(image_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)

# Biến ghi video
is_recording = False
video_writer = None


def auto_capture_photos(track_id, frame):
    global photos_taken
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f'ID_{track_id}_{current_time}'
    directory_path = os.path.join(image_folder, folder_name)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    if photos_taken < photos_to_take:
        photo_name = f'{track_id}_{photos_taken + 1}_{current_time}.png'
        photo_path = os.path.join(directory_path, photo_name)
        cv2.imwrite(photo_path, frame)
        photos_taken += 1
    elif photos_taken >= photos_to_take:
        photos_taken = 0  # Reset counter after taking 10 photos

def set_yaw_vel(x_error, p_x_error, pid_x, last_update_time):
    current_time = time.time()
    if current_time - last_update_time < 0.1:
        return 0  # Không cập nhật vận tốc nếu chưa đủ 0.1 giây
    p_term = pid_x[0] * x_error
    i_term = pid_x[1] * (x_error - p_x_error)
    speed_x = p_term + i_term
    speed_x = int(np.clip(speed_x, -100, 100))
    return speed_x

def set_y_vel(y_error, p_y_error, pid_y, last_update_time):
    current_time = time.time()
    if current_time - last_update_time < 0.1:
        return 0  #  Tương tự như trên
    p_term = pid_y[0] * y_error
    i_term = pid_y[1] * (y_error - p_y_error)
    speed_y = p_term + i_term
    speed_y = int(np.clip(speed_y, -100, 100))
    return speed_y
#
def set_z_vel(area_error, p_area_error, pid_z, last_update_time):
    current_time = time.time()
    if current_time - last_update_time < 0.1:
        return 0  # Tương tự như trên
    p_term = pid_z[0] * area_error
    i_term = pid_z[1] * (area_error - p_area_error)
    speed_z = p_term + i_term
    speed_z = int(np.clip(speed_z, -100, 100))
    return speed_z

while True:
    frame_start_time = time.time()
    frame_read = me.get_frame_read()
    annotated_frame = frame_read.frame.copy()

    if frame_read.frame is not None:
        frame = cv2.resize(frame_read.frame, (video_w, video_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.track(frame, tracker='bytetrack.yaml', persist=True, device='cuda', iou=0.5, conf=0.5, classes=0, verbose=False)

        boxes = []
        track_ids = []
        class_ids = []

        if results[0].boxes and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.cpu().numpy()
        else:
            notification_text = "Target not found"
            text_coordinates = (int(370), int(370))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_color = (0, 0, 255)
            line_type = 2
            cv2.putText(frame, notification_text, text_coordinates, font, font_scale, font_color, line_type)
            me.up_down_velocity = 0
            me.yaw_velocity = 0
            me.for_back_velocity = 0
            me.left_right_velocity = 0

        annotated_frame = results[0].plot()

        if id_tracking[0] == -1:
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x, y, w, h = box.cpu().numpy()
                area_bbox = int(w * h)
                ideal_areas[track_id] = area_bbox
        elif id_tracking[0] != -1:
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if id_tracking[0] == track_id:
                    x, y, w, h = box.cpu().numpy()
                    center_x_bbox = int(x)
                    center_y_bbox = int(y)
                    area_bbox = int(w * h)

                    x_error = center_x_bbox - center_x_frame
                    y_error = center_y_bbox - center_y_frame
                    z_error = area_bbox - ideal_areas[id_tracking[0]]
                    if z_error > 0:
                        z_error = z_error - dead_zone
                    elif z_error < 0:
                        z_error = z_error + dead_zone
                    else:
                        z_error = 0

                    me.for_back_velocity = set_z_vel(z_error, z_error_new, pid_z, last_update_time)
                    me.up_down_velocity = set_y_vel(y_error, y_error_new, pid_y, last_update_time)
                    me.yaw_velocity = set_yaw_vel(x_error, x_error_new, pid_x, last_update_time)

                    x_error_new = x_error
                    y_error_new = y_error
                    z_error_new = z_error

                    text_to_display = f"[{x_error} {y_error} {z_error}]"
                    cv2.putText(annotated_frame, text_to_display, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    cv2.circle(annotated_frame, (center_x_bbox, center_y_bbox), 2, (0, 255, 255), 2)
                    cv2.arrowedLine(annotated_frame, (center_x_bbox, center_y_bbox), (int(center_x_frame), int(center_y_frame)), (0, 255, 255), 2)

    if id_tracking[0] != -1 and id_tracking[0] == track_id:
        auto_capture_photos(id_tracking[0], annotated_frame)
    # Hiển thi FPS
    frame_end_time = time.time()
    times.append(frame_end_time - frame_start_time)
    if len(times) > 100:
        times = times[-100:]
    avg_frame_time = np.mean(times)
    actual_fps = 1 / avg_frame_time if avg_frame_time > 0 else 0
    fps_text = f"FPS: {actual_fps:.2f}"
    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Hiển thị pin của DJI Tello
    cv2.putText(annotated_frame, f"Drone Battery = {me.get_battery()}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị thời gian hiện tại
    cv2.putText(annotated_frame, formatted_datetime, (900,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    # Nếu không có ID tracking, reset tất cả vận tốc về 0
    if id_tracking[0] == -1:
        me.for_back_velocity = 0
        me.left_right_velocity = 0
        me.up_down_velocity = 0
        me.yaw_velocity = 0


    #Lấy chiều cao của drone
    drone_height = me.get_height()
    height_test = f"Drone Height: {drone_height} cm"
    cv2.putText(annotated_frame, height_test, (900, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Object Tracking", annotated_frame)

    # Kiểm tra nhập từ bàn phím
    key = cv2.waitKey(1) & 0xFF

    # Bắt đầu hoặc dừng ghi video khi nhấn 'v'
    if key == ord('v'):
        if not is_recording:
            # Tạo mới video writer khi bắt đầu ghi
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Lấy thời gian hiện tại
            video_filename = os.path.join(video_folder, f'output_video_{current_time}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Sử dụng codec cho mp4
            video_writer = cv2.VideoWriter(video_filename, fourcc, 40.0, (video_w, video_h))
            is_recording = True
        else:
            # Dừng ghi khi đang ghi
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            is_recording = False

    # Ghi video nếu đang ghi
    if is_recording:
        video_writer.write(annotated_frame)

    # Chụp ảnh khi nhấn 'c'
    if key == ord('c'):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Lấy thời gian hiện tại
        screenshot_filename = os.path.join(image_folder, f'screenshot_{current_time}.png')  # Tạo tên file mới
        cv2.imwrite(screenshot_filename, annotated_frame)  # Lưu ảnh

    me.send_rc_control(me.left_right_velocity, -me.for_back_velocity, -me.up_down_velocity, me.yaw_velocity)

    # Nhấn q để thoát khỏi chương trình
    if key == ord("q"):
        if is_recording:
            video_writer.release()
            video_writer = None
        me.streamoff()
        cv2.destroyAllWindows()
        me.land()
        break