import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ============ 1. 通用卷积函数（单通道图像）============

def manual_conv2d(image, kernel):

    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    # 零填充
    padded = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w), dtype=np.float32)
    padded[pad_h:pad_h+img_h, pad_w:pad_w+img_w] = image.astype(np.float32)

    # 卷积结果
    result = np.zeros_like(image, dtype=np.float32)

    # 翻转卷积核（严格的卷积运算）
    flipped_kernel = np.flipud(np.fliplr(kernel.astype(np.float32)))

    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i+k_h, j:j+k_w]
            result[i, j] = np.sum(region * flipped_kernel)

    return result

# ============ 2. Sobel 算子实现 ============

def sobel_edge_detection(gray):

    # Sobel 核（可直接在报告里写出来）
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    grad_x = manual_conv2d(gray, sobel_x)
    grad_y = manual_conv2d(gray, sobel_y)

    # 梯度幅值
    mag = np.sqrt(grad_x**2 + grad_y**2)

    # 归一化到 0-255
    mag = mag / (mag.max() + 1e-6) * 255.0
    sobel_mag = mag.astype(np.uint8)
    return sobel_mag

# ============ 3. 给定卷积核滤波 ============

def custom_filter(gray):

    custom_kernel = np.array([[ 1, 0,  -1],
                              [2,  0, -2],
                              [ 1, 0,  -1]], dtype=np.float32)

    filtered = manual_conv2d(gray, custom_kernel)

    # 为了显示，把结果归一化到 0-255
    min_val, max_val = filtered.min(), filtered.max()
    normalized = (filtered - min_val) / (max_val - min_val + 1e-6) * 255.0
    return normalized.astype(np.uint8)

# ============ 4. 手写颜色直方图计算 ============
def manual_color_hist_16(image):

    h, w, _ = image.shape
    bins = 16
    bin_size = 256 // bins  # 每个区间宽度 = 16

    b_hist = np.zeros(bins, dtype=np.int64)
    g_hist = np.zeros(bins, dtype=np.int64)
    r_hist = np.zeros(bins, dtype=np.int64)

    # 遍历像素统计频数
    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]

            b_hist[min(b // bin_size, bins - 1)] += 1
            g_hist[min(g // bin_size, bins - 1)] += 1
            r_hist[min(r // bin_size, bins - 1)] += 1

    # 生成区间标签，如 "0-15", "16-31", ...
    bin_ranges = [f"{i*bin_size}-{(i+1)*bin_size-1}" for i in range(bins)]

    return b_hist, g_hist, r_hist, bin_ranges
def plot_rgb_hist_grouped(b_hist, g_hist, r_hist, bin_ranges, save_path="color_hist.png"):
    bins = len(bin_ranges)
    x = np.arange(bins)  # 每个区间的位置

    width = 0.25  # 三个柱子并排宽度

    plt.figure(figsize=(12, 6))

    # 三个通道的柱状图并排显示
    plt.bar(x - width, b_hist, width=width, color='blue', label='Blue')
    plt.bar(x,         g_hist, width=width, color='green', label='Green')
    plt.bar(x + width, r_hist, width=width, color='red', label='Red')

    # 设置 X 轴为区间标签
    plt.xticks(x, bin_ranges, rotation=45)

    plt.xlabel("Pixel Value Range (Bins)")
    plt.ylabel("Frequency")
    plt.title("RGB Histogram (16 Bins)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============ 5. 手写 GLCM 纹理特征 ============

def compute_glcm_features(gray, levels=8, distance=1, angle=0):


    # 1) 量化灰度到 0 ~ levels-1
    # 例如 levels=8，则每 32 个灰度级映射到一个等级
    gray = gray.astype(np.uint8)
    quantized = (gray / (256 // levels)).astype(np.int32)


    quantized[quantized >= levels] = levels - 1  # 防止舍入误差到 levels

    h, w = quantized.shape

    # 2) 初始化 GLCM
    glcm = np.zeros((levels, levels), dtype=np.float64)

    # 3) 只考虑 (i, j+distance) 的水平对
    for i in range(h):
        for j in range(w - distance):
            g1 = quantized[i, j]
            g2 = quantized[i, j + distance]
            glcm[g1, g2] += 1.0

    # 4) 归一化为概率
    if glcm.sum() > 0:
        glcm = glcm / glcm.sum()

    # 5) 计算纹理特征
    contrast = 0.0
    energy = 0.0
    homogeneity = 0.0
    entropy = 0.0

    for i in range(levels):
        for j in range(levels):
            p = glcm[i, j]
            if p <= 0:
                continue
            contrast += (i - j) ** 2 * p
            energy += p ** 2
            homogeneity += p / (1.0 + abs(i - j))
            entropy += -p * math.log2(p)

    features = np.array([contrast, energy, homogeneity, entropy], dtype=np.float64)
    return features, glcm

# ============ 6. 主流程 ============

def main():
    # 你自己的照片路径
    image_path = "Lenna.jpg"

    # 读图（BGR）
    img = cv2.imread(image_path)
    if img is None:
        print("读取图像失败，请检查路径:", image_path)
        return

    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Sobel 边缘检测
    sobel_img = sobel_edge_detection(gray)
    cv2.imwrite("homework/result_sobel.png", sobel_img)

    # 2) 给定卷积核滤波
    custom_filtered = custom_filter(gray)
    cv2.imwrite("result_custom_filter.png", custom_filtered)

    # 3) 手写颜色直方图
    b_hist, g_hist, r_hist, bin_ranges = manual_color_hist_16(img)
    plot_rgb_hist_grouped(b_hist, g_hist, r_hist, bin_ranges, "result_color_hist.png")

    # 4) 手写 GLCM 纹理特征
    texture_features, glcm = compute_glcm_features(gray, levels=8)
    print("纹理特征 [contrast, energy, homogeneity, entropy] =")
    print(texture_features)

    # 保存纹理特征到 npy
    np.save("texture_features.npy", texture_features)
    np.save("glcm_matrix.npy", glcm)

    cv2.imshow("Original", img)
    cv2.imshow("Gray", gray)
    cv2.imshow("Sobel", sobel_img)
    cv2.imshow("Custom Filter", custom_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
