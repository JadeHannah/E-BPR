import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免交互显示
import matplotlib.pyplot as plt
import os

# ========== 辅助函数：边界提取 ==========
def mask_to_boundary(mask, dilation_ratio=0.005):
    """
    将二值 mask 转为边界 mask。
    Args:
        mask: np.ndarray, shape=(H, W), 值为0或1
        dilation_ratio: 边界厚度比例，例如0.005表示边界宽度≈图像对角线的0.5%
    Returns:
        boundary: np.ndarray, 二值边界图
    """
    h, w = mask.shape
    diag_len = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * diag_len)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    boundary = dilated - eroded
    return boundary

# ========== 辅助函数：BIoU 计算 ==========
def compute_biou(pred_mask, gt_mask, dilation_ratio=0.005, combine_with_iou=True):
    """
    计算 Boundary IoU (BIoU)
    Args:
        pred_mask: 预测的二值掩膜 (numpy array, 0/1)
        gt_mask:   GT 二值掩膜 (numpy array, 0/1)
        dilation_ratio: 边界宽度比例
        combine_with_iou: 若为True，返回 min(IoU, BIoU)，否则仅返回边界IoU
    Returns:
        final_iou, iou, biou
    """
    # 普通 IoU
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = inter / union if union > 0 else 0.0

    # 边界 IoU
    pred_boundary = mask_to_boundary(pred_mask, dilation_ratio)
    gt_boundary = mask_to_boundary(gt_mask, dilation_ratio)
    inter_b = np.logical_and(pred_boundary, gt_boundary).sum()
    union_b = np.logical_or(pred_boundary, gt_boundary).sum()
    biou = inter_b / union_b if union_b > 0 else 0.0

    final_iou = min(iou, biou) if combine_with_iou else biou
    return final_iou, iou, biou

# ========== 掩膜加载函数 ==========
def load_mask(path):
    """
    读取 mask 文件，支持 .png 和 .npy
    返回二值化的 numpy array (0/1)
    """
    if path.endswith('.png'):
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)
    elif path.endswith('.npy'):
        mask = np.load(path)
        mask = (mask > 0).astype(np.uint8)
    else:
        raise ValueError("不支持的文件类型: 仅支持 .png 或 .npy")
    return mask

# ========== 主可视化函数 ==========
def visualize_biou(pred_path, gt_path, dilation_ratio=0.005):
    """
    加载预测与GT掩膜，计算并显示 BIoU
    """
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT 文件不存在: {gt_path}")

    pred_mask = load_mask(pred_path)
    gt_mask = load_mask(gt_path)

    # 计算 BIoU
    final_iou, mask_iou, boundary_iou = compute_biou(pred_mask, gt_mask, dilation_ratio)
    print(f"\n=== Boundary IoU Evaluation ===")
    print(f"GT:   {os.path.basename(gt_path)}")
    print(f"Pred: {os.path.basename(pred_path)}")
    print(f"Mask IoU:      {mask_iou:.4f}")
    print(f"Boundary IoU:  {boundary_iou:.4f}")
    print(f"Final (min):   {final_iou:.4f}\n")

    # 可视化
    gt_boundary = mask_to_boundary(gt_mask, dilation_ratio)
    pred_boundary = mask_to_boundary(pred_mask, dilation_ratio)

    H, W = gt_mask.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[gt_mask > 0] = [0, 255, 0]       # GT: 绿色
    overlay[pred_mask > 0] = [255, 0, 0]     # Pred: 蓝色
    overlay[np.logical_and(gt_mask > 0, pred_mask > 0)] = [255, 255, 0]  # 重叠: 黄色

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 4, 1); plt.imshow(gt_mask, cmap='gray'); plt.title('GT Mask')
    plt.subplot(1, 4, 2); plt.imshow(pred_mask, cmap='gray'); plt.title('Pred Mask')
    plt.subplot(1, 4, 3); plt.imshow(gt_boundary + pred_boundary, cmap='hot'); plt.title('Boundaries')
    plt.subplot(1, 4, 4); plt.imshow(overlay[..., ::-1]); plt.title(f'Overlay\nBIoU={boundary_iou:.3f}')
    plt.tight_layout()
    plt.savefig("biou_result.png", dpi=150)
    print("BIoU 可视化结果已保存为 biou_result.png")


# ========== 示例调用 ==========
if __name__ == "__main__":
    # 示例：你可以替换为自己的文件路径
    gt_path = r"D:\hannahtrain\ultralytics\ultralytics\danzhantumask\530_jpg.rf.23b5ecaf2d0dcf49d4692c2e903f5879_inst5.png"         # GT 掩膜路径
    pred_path = r"D:\hannahtrain\ultralytics\ultralytics\prepare_result\masks\530_jpg.rf.23b5ecaf2d0dcf49d4692c2e903f5879_3_paddy field.png"     # 预测掩膜路径
    visualize_biou(pred_path, gt_path, dilation_ratio=0.005)
