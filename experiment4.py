import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont


# =========================
# COCO 类别定义
# =========================
COCO_CATEGORIES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

BICYCLE_ID = COCO_CATEGORIES.index("bicycle")


# =========================
# 加载 COCO 预训练模型
# =========================
def load_model(device):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess


# =========================
# 共享单车检测
# =========================
@torch.no_grad()
def detect_bicycles(model, preprocess, img_pil, device, score_thr):
    x = preprocess(img_pil).to(device)
    output = model([x])[0]

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    keep = (labels == BICYCLE_ID) & (scores >= score_thr)
    boxes = boxes[keep]
    scores = scores[keep]

    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]

    results = []
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in b]
        results.append({
            "label": "bicycle",
            "score": float(s),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [x1, y1, x2 - x1, y2 - y1]
        })

    return results


# =========================
# 绘制检测框
# =========================
def draw_boxes(img_pil, results, out_path):
    img = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    for r in results:
        x1, y1, x2, y2 = r["bbox_xyxy"]
        score = r["score"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 18)),
                  f"bicycle {score:.2f}", fill=(255, 0, 0), font=font)

    img.save(out_path)


# =========================
# 主函数（IDE 直接运行）
# =========================
def main():
    # ===== 在这里直接修改图片路径即可 =====
    IMAGE_PATH = "campus_bike.jpg"     # 输入图片
    OUT_IMG = "bike_result.jpg"        # 输出画框图
    OUT_JSON = "bike_result.json"      # 输出检测结果
    SCORE_THR = 0.6                    # 置信度阈值

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    img_pil = Image.open(IMAGE_PATH).convert("RGB")

    model, preprocess = load_model(device)
    results = detect_bicycles(model, preprocess, img_pil, device, SCORE_THR)

    print(f"检测到 bicycle 数量：{len(results)}")
    for i, r in enumerate(results, 1):
        x1, y1, x2, y2 = r["bbox_xyxy"]
        print(f"[{i}] score={r['score']:.3f}, "
              f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

    # 保存 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "image": IMAGE_PATH,
            "task": "shared_bike_detection",
            "model": "Faster R-CNN ResNet50-FPN (COCO pretrained)",
            "score_threshold": SCORE_THR,
            "detections": results
        }, f, ensure_ascii=False, indent=2)

    # 保存结果图
    draw_boxes(img_pil, results, OUT_IMG)

    print("结果已保存：", OUT_IMG, "和", OUT_JSON)


if __name__ == "__main__":
    main()
