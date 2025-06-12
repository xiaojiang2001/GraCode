import cv2
import numpy as np
from ultralytics import YOLO


def detect_objects_and_temperature(image_path, show_label=True, min_area=100):
    """
    先进行行人和车辆检测，然后检测高温区域并标注

    参数:
        image_path: 红外图像路径
        temp_threshold: 温度阈值(高于此值视为高温)
        show_label: 是否显示目标类别标签
        min_area: 最小区域面积(小于此值的高温区域将被忽略)
    """
    # 加载YOLOv8模型
    model = YOLO('yolov8n.pt')

    # 读取RGB图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像，请检查路径")
        return

    # 分离通道，使用红色通道作为温度信息
    b, g, r = cv2.split(img)
    temp_map = r  # 使用红色通道作为温度图

    # 目标检测
    results = model(img)
    detected_img = img.copy()

    # 绘制检测到的目标
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 只关注人和车
            if box.cls.cpu().numpy()[0] in [0, 2, 3, 5, 7]:  # person(0), car(2), motorcycle(3), bus(5), truck(7)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # 绘制边界框
                cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                
                # 根据用户选择显示标签
                if show_label:
                    class_name = model.names[cls_id]
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(detected_img, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
    # 显示结果
    cv2.imshow("目标检测", detected_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detected_img

# 使用示例
if __name__ == "__main__":
    # 参数说明:
    # 1. 图像路径
    # 2. 温度阈值(根据实际情况调整)
    # 3. 是否显示目标类别标签
    # 4. 最小区域面积(可选)
    detect_objects_and_temperature("./M3FD/ours/00290.png", show_label=True, min_area=50)