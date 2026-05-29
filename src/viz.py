import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def _get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _annotate(base_img, top3):
    """Composite a semi-transparent prediction label block onto a PIL RGBA image."""
    overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(20)

    lines = ["Top-3 predictions:"]
    for i, (_, label, prob) in enumerate(top3, start=1):
        lines.append(f"{i}. {label}: {prob:.2%}")
    text = "\n".join(lines)

    padding = 8
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.multiline_textsize(text, font=font)

    x0, y0 = 10, 10
    draw.rectangle([x0, y0, x0 + tw + padding * 2, y0 + th + padding * 2],
                   fill=(0, 0, 0, 160))
    draw.multiline_text((x0 + padding, y0 + padding), text,
                        fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base_img, overlay)


def save_overlay(heatmap, orig_img, top3, out_path='figures/cat_gradcam_overlay.png'):
    """
    Colorize heatmap with jet colormap, blend with original image at alpha=0.4,
    annotate top-3 predictions, and save to out_path.
    """
    heatmap_rgba = np.uint8(plt.cm.jet(heatmap) * 255)
    heatmap_img = Image.fromarray(heatmap_rgba).convert('RGBA')
    heatmap_img = heatmap_img.resize(orig_img.size, resample=Image.BILINEAR)
    superimposed = Image.blend(orig_img.convert('RGBA'), heatmap_img, alpha=0.4)
    annotated = _annotate(superimposed, top3)
    annotated.save(out_path)
    print(f'Saved overlay → {out_path}')


def save_heatmap(heatmap, orig_img, top3, out_path='figures/cat_heatmap.png'):
    """
    Save a grayscale heatmap resized to the original image dimensions,
    annotated with top-3 predictions.
    """
    heatmap_gray = Image.fromarray(np.uint8(heatmap * 255)).resize(orig_img.size)
    annotated = _annotate(heatmap_gray.convert('RGBA'), top3)
    annotated.save(out_path)
    print(f'Saved heatmap  → {out_path}')
