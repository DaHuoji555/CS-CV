import cv2
import numpy as np

# ========== 1. ROI 区域 ==========
def region_of_interest(img):
    h, w = img.shape[:2]

    # 你调整后的梯形 ROI
    roi_points = np.array([
        [int(w * 0),    int(h * 0.75)],  # 左下
        [int(w * 1),    int(h * 0.75)],  # 右下
        [int(w * 0.58), int(h * 0.45)],  # 右上
        [int(w * 0.47), int(h * 0.45)],  # 左上
    ], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [roi_points], 255)

    masked = cv2.bitwise_and(img, mask)
    return masked, roi_points


# ========== 2. 拟合左右车道线 ==========
def average_lines(img, lines):
    left_lines = []
    right_lines = []
    height = img.shape[0]

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < -0.5:        # 左车道线（负斜率）
            left_lines.append((slope, intercept))
        elif slope > 0.5:       # 右车道线（正斜率）
            right_lines.append((slope, intercept))

    if not left_lines and not right_lines:
        return None

    def make_points(slope, intercept):
        y1 = height
        y2 = int(height * 0.48)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    results = []
    if left_lines:
        slope, b = np.mean(left_lines, axis=0)
        results.append(make_points(slope, b))

    if right_lines:
        slope, b = np.mean(right_lines, axis=0)
        results.append(make_points(slope, b))

    return results


# ========== 3. 主流程 ==========
def detect_lane(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("读取失败：", image_path)
        return

    h, w = img.shape[:2]

    # 显示原图
    cv2.imshow("1 - Original", img)

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 边缘
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow("2 - Canny Edges", edges)

    # ROI
    cropped_edges, roi_points = region_of_interest(edges)
    cv2.imshow("3 - ROI Masked", cropped_edges)

    # 显示 ROI 可视化
    roi_visual = img.copy()
    cv2.polylines(roi_visual, [roi_points], True, (0, 255, 0), 3)
    cv2.imshow("4 - ROI Visualization", roi_visual)
    cv2.imwrite("roi_visualize.png", roi_visual)

    # 霍夫直线检测
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=60,
        maxLineGap=80
    )

    # 显示霍夫检测结果
    hough_img = img.copy()
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("5 - Hough Raw Lines", hough_img)
    cv2.imwrite("hough_raw.png", hough_img)

    # 计算平均左右两条车道线
    averaged = average_lines(img, lines)

    output = img.copy()
    if averaged:
        for x1, y1, x2, y2 in averaged:
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 8)

    cv2.imshow("6 - Final Lane", output)
    cv2.imwrite("lane_result.png", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ========== 运行 ==========

detect_lane("road.png")
