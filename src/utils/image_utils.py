from PIL import Image


def resize_for_sd(image: Image.Image, size: int = 512) -> Image.Image:
    """Resize image to square by center-cropping then resizing."""
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    cropped = image.crop((left, top, left + min_dim, top + min_dim))
    return cropped.resize((size, size), Image.LANCZOS)


def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")
