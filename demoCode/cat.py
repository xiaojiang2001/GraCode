import cv2
import numpy as np
import os


def image_concat(image_paths, direction='horizontal', resize=True, target_size=None):
    """
    拼接多张图片
    
    参数:
        image_paths: 图像路径列表
        direction: 拼接方向，'horizontal'（水平）或'vertical'（垂直）
        resize: 是否调整图像大小使其一致
        target_size: 目标大小，格式为(width, height)。如果为None且resize=True，则使用第一张图片的大小
    """
    # 读取所有图像
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"无法读取图像: {path}")
            continue
        images.append(img)
    
    if not images:
        print("没有有效的图像可以拼接")
        return None
    
    # 确定目标大小
    if resize:
        if target_size is None:
            target_size = (images[0].shape[1], images[0].shape[0])  # (width, height)
        
        # 调整所有图像大小
        resized_images = []
        for img in images:
            resized = cv2.resize(img, target_size)
            resized_images.append(resized)
    else:
        resized_images = images
    
    # 根据方向拼接图像
    if direction == 'horizontal':
        return np.hstack(resized_images)
    else:  # vertical
        return np.vstack(resized_images)


def stitch_images(image_paths):
    """
    将多张图片拼接成全景图
    
    参数:
        image_paths: 图像路径列表（图像应该有重叠区域）
    返回:
        全景拼接后的图像
    """
    # 读取所有图像
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"无法读取图像: {path}")
            continue
        images.append(img)
    
    if len(images) < 2:
        print("需要至少两张图片进行拼接")
        return None

    # 创建Stitcher对象
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    
    # 进行图像拼接
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        print("全景图拼接成功")
        return panorama
    else:
        print("全景图拼接失败")
        return None


def stitch_images_with_features(image_paths):
    """
    使用特征匹配的方式拼接全景图（更详细的控制）
    
    参数:
        image_paths: 图像路径列表（图像应该有重叠区域）
    返回:
        全景拼接后的图像
    """
    # 读取所有图像
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"无法读取图像: {path}")
            continue
        images.append(img)
    
    if len(images) < 2:
        print("需要至少两张图片进行拼接")
        return None

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()
    
    # 初始化特征匹配器
    bf = cv2.BFMatcher()
    
    def match_and_stitch(img1, img2):
        # 检测特征点和描述符
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
        
        # 特征匹配
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 应用比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > 10:
            # 获取匹配点的坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # 获取图像尺寸
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                
                # 计算变换后的图像范围
                pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
                pts2 = cv2.perspectiveTransform(pts1, H)
                pts = np.concatenate((pts2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
                
                # 计算输出图像的尺寸
                [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
                [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
                t = [-xmin, -ymin]
                
                # 创建平移矩阵
                Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
                
                # 对图像进行变换
                result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
                
                # 将第二张图像复制到结果中
                result[t[1]:h2+t[1], t[0]:w2+t[0]] = img2
                
                return result
        
        return None

    # 逐步拼接所有图像
    result = images[0]
    for i in range(1, len(images)):
        result = match_and_stitch(result, images[i])
        if result is None:
            print(f"在拼接第{i+1}张图片时失败")
            return None
    
    return result


if __name__ == "__main__":
    # 示例用法
    image_dir = "./cat"  # 图片所在目录
    image_paths = [os.path.join(image_dir, f"{i}.jpg") for i in range(1, 4)]  # 获取1.jpg到3.jpg的路径

    # # 水平拼接
    # result_h = image_concat(image_paths, direction='horizontal', resize=True, target_size=(400, 300))
    # if result_h is not None:
    #     cv2.imshow('Horizontal Concatenation', result_h)
        # cv2.imwrite('horizontal_result.jpg', result_h)  # 保存结果
    
    # 垂直拼接
    # result_v = image_concat(image_paths, direction='vertical', resize=True, target_size=(400, 300))
    # if result_v is not None:
    #     cv2.imshow('Vertical Concatenation', result_v)
    #     cv2.imwrite('vertical_result.jpg', result_v)  # 保存结果

    # 使用OpenCV内置的Stitcher
    print("使用OpenCV Stitcher进行拼接...")
    panorama1 = stitch_images(image_paths)
    panorama1 = cv2.resize(panorama1, (900, 400))  # 调整全景图大小
    if panorama1 is not None:
        cv2.imshow('Panorama (OpenCV Stitcher)', panorama1)
        cv2.imwrite('panorama_opencv.jpg', panorama1)

    # 使用特征匹配的方式
    print("使用特征匹配方式进行拼接...")
    panorama2 = stitch_images_with_features(image_paths)
    panorama2 = cv2.resize(panorama2, (900, 400))  # 调整全景图大小
    if panorama2 is not None:
        cv2.imshow('Panorama (Feature Matching)', panorama2)
        cv2.imwrite('panorama_features.jpg', panorama2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()