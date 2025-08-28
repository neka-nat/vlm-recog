import hashlib
import colorsys
from typing import Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vlm_recog.models import DetectedItems
from vlm_recog import bbox_utils


def draw_detections(
    image: Image.Image,
    detections: DetectedItems,
    *,
    draw_mask: bool = True,
    draw_box: bool = True,
    mask_alpha: float = 0.45,
    box_thickness: int = 2,
    label_bg_alpha: int = 160,
    threshold: float | None = None,
    color_map: Dict[str, Tuple[int, int, int]] | None = None,
    inplace: bool = False,
    font_path: str | None = None,
    font_size: int = 14,
) -> Image.Image:
    """
    物体検出結果(2D bbox, セグメンテーションマスク, ラベル)を PIL.Image に描画する。

    Args:
        image: 描画対象の画像 (PIL.Image)
        detections: DetectedItem のリスト
        draw_mask: True のときマスクを半透明で重ねる
        draw_box: True のときバウンディングボックスを描画
        mask_alpha: マスクの不透明度(0〜1)
        box_thickness: バウンディングボックスの線幅(px)
        label_bg_alpha: ラベルの背景(黒)の不透明度(0〜255)
        threshold: マスクが確率マップ/0-255 の場合に二値化する閾値(Noneなら自動)
        color_map: ラベル→RGB の色辞書(未指定ならラベルに基づき自動生成・固定)
        inplace: True のとき元画像に直接描画、False ならコピーを返す
        font_path: ラベル描画に使う TTF フォントパス(日本語ラベルなど必要なら指定)
        font_size: ラベル文字サイズ

    Returns:
        描画済みの PIL.Image
    """
    if not inplace:
        img = image.copy()
    else:
        img = image

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    W, H = img.size

    # カラーマップが無ければラベルから決定論的に生成
    if color_map is None:
        labels = sorted(set(d.label for d in detections))
        color_map = {lab: _label_to_color(lab) for lab in labels}

    # フォント
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # マスクはオーバーレイ合成、枠とテキストは ImageDraw
    draw = ImageDraw.Draw(img, "RGBA")

    for det in detections:
        x1, y1, x2, y2 = bbox_utils.clip_box(det.box_2d, W, H)
        if x1 >= x2 or y1 >= y2:
            continue  # 完全に画面外

        color = tuple(color_map.get(det.label, (0, 255, 0)))  # RGB
        rgba = (*color, 255)

        if draw_mask and det.segmentation_mask is not None:
            mask_bool = _to_bool_mask(det.segmentation_mask, threshold=threshold)

            # マスクのサイズが画像全体か、bboxか、その他かで処理
            mh, mw = mask_bool.shape[:2]
            if (mw, mh) == (W, H):
                mask_full = mask_bool
            elif (mw, mh) == (x2 - x1, y2 - y1):
                mask_full = np.zeros((H, W), dtype=bool)
                mask_full[y1:y2, x1:x2] = mask_bool
            else:
                # bbox にリサイズして敷く（最近傍で二値性を維持）
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    resized = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L").resize(
                        (x2 - x1, y2 - y1), resample=Image.NEAREST
                    )
                    mask_full = np.zeros((H, W), dtype=bool)
                    mask_full[y1:y2, x1:x2] = (np.array(resized) > 127)
                else:
                    mask_full = None

            if mask_full is not None:
                overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                # 指定色 + alpha で塗る
                alpha_val = int(255 * max(0.0, min(1.0, mask_alpha)))
                arr = np.zeros((H, W, 4), dtype=np.uint8)
                arr[mask_full] = np.array([*color, alpha_val], dtype=np.uint8)
                overlay = Image.fromarray(arr, mode="RGBA")
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img, "RGBA")  # 再アタッチ

        if draw_box:
            _draw_rect(draw, (x1, y1, x2, y2), rgba, thickness=box_thickness)

        text = det.label
        if text:
            # テキスト背景ボックス
            tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), text, font=font)
            bg = (0, 0, 0, max(0, min(255, label_bg_alpha)))
            # 画像外に出ないように多少補正
            tx1 = max(0, tx1)
            ty1 = max(0, ty1)
            tx2 = min(W, tx2)
            ty2 = min(H, ty2)
            draw.rectangle([tx1, ty1, tx2, ty2], fill=bg)
            draw.text((tx1, ty1), text, fill=(255, 255, 255, 255), font=font)

    return img


def _label_to_color(label: str) -> Tuple[int, int, int]:
    """ラベル文字列から安定的な色(RGB)を生成（見やすい彩度・明度に固定）。"""
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 360
    s, v = 0.75, 1.0
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _draw_rect(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], color_rgba, thickness: int = 2):
    x1, y1, x2, y2 = box
    for t in range(thickness):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color_rgba)


def _to_bool_mask(arr: np.ndarray, threshold: float | None) -> np.ndarray:
    """
    入力配列を bool マスクに正規化:
      - 既に bool → そのまま
      - 浮動小数 → threshold(既定0.5)で二値化
      - 整数/8bit → threshold(既定>0)で二値化
    """
    if arr.dtype == np.bool_:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        thr = 0.5 if threshold is None else float(threshold)
        return arr >= thr
    # 整数系
    thr = 0.0 if threshold is None else float(threshold)
    return arr.astype(np.float32) > thr
