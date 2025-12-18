import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
import os

# =========================
# 1. CNN 模型定义
# =========================
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =========================
# 2. 训练 MNIST
# =========================
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=True
    )

    model = DigitCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("开始训练 MNIST 模型...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), "digit_cnn.pth")
    print("模型训练完成，已保存为 digit_cnn.pth")

# =========================
# 3. MNIST风格数字预处理（核心改动）
# =========================
def preprocess_to_mnist_style(digit):
    """
    输入：二值化后的单个数字（白字黑底，numpy）
    输出：28x28，接近MNIST风格
    """
    h, w = digit.shape

    # 等比缩放，使最大边为20
    scale = 20.0 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    digit = cv2.resize(digit, (new_w, new_h))

    # 放到28x28画布中心
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    return canvas

# =========================
# 4. 学号图片识别
# =========================
def recognize_student_id(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DigitCNN().to(device)
    model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
    model.eval()

    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"未找到图片: {image_path}")
        return

    # 2. 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 二值化（白字黑底 → 黑字白底）
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. 形态学操作（轻量即可）
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 保存中间结果（方便写报告）
    cv2.imwrite("1_gray.jpg", gray)
    cv2.imwrite("2_binary.jpg", binary)
    cv2.imwrite("3_processed.jpg", processed)

    # 5. 轮廓分割
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digit_info = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 150:  # 去噪
            continue
        digit_info.append({"x": x, "y": y, "w": w, "h": h})

    # 按x从左到右
    digit_info.sort(key=lambda d: d["x"])

    student_id = ""
    result_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # 6. 逐个数字识别
    for i, d in enumerate(digit_info):
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        roi = processed[y:y + h, x:x + w]

        # MNIST风格预处理（关键）
        digit_28 = preprocess_to_mnist_style(roi)
        digit_28 = digit_28 / 255.0

        digit_tensor = torch.tensor(digit_28, dtype=torch.float32)
        digit_tensor = digit_tensor.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(digit_tensor)
            num = pred.argmax(dim=1).item()
            conf = F.softmax(pred, dim=1)[0][num].item()

        student_id += str(num)

        print(f"数字{i+1}: 识别为 {num}, 置信度 {conf:.2f}")

        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_img, f"{num}({conf:.2f})",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    cv2.imwrite("4_result.jpg", result_img)

    print("\n" + "=" * 50)
    print(f"识别出的学号为：{student_id}")
    print("=" * 50)

    return student_id

# =========================
# 5. 主程序入口
# =========================
if __name__ == "__main__":
    if not os.path.exists("digit_cnn.pth"):
        train_model()
    else:
        print("检测到已训练模型，跳过训练")

    recognize_student_id("student_id.jpg")
