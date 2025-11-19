import argparse
import os
import sys
import types
import glob

import torch
import cv2

# ============================================================================
# Fix deprecated torchvision import expected by basicsr
# ============================================================================
try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale  # old path
except Exception:
    from torchvision.transforms.functional import rgb_to_grayscale  # new path
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
# ============================================================================


def build_upsampler(scale: int) -> RealESRGANer:
    """Load RRDBNet + RealESRGANer with auto-downloaded weights."""
    if scale == 4:
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
            "RealESRGAN_x4plus.pth"
        )
    elif scale == 2:
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0.4/"
            "RealESRGAN_x2plus.pth"
        )
    else:
        raise ValueError("Scale must be 2 or 4.")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_url,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )

    return upsampler


def process_single_image(input_path: str, output_path: str, upsampler, scale: int):
    """Upscale a single image."""
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[!] Failed to read: {input_path}")
        return False

    with torch.no_grad():
        upscaled, _ = upsampler.enhance(img, outscale=scale)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, upscaled)
    return True


def process_folder(input_dir: str, output_dir: str, upsampler, scale: int):
    """Upscale every image in a folder. Supports png/jpg/jpeg."""
    supported = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for s in supported:
        files.extend(glob.glob(os.path.join(input_dir, s)))

    if not files:
        print("[!] No images found in folder.")
        return

    print(f"[+] Found {len(files)} images")
    os.makedirs(output_dir, exist_ok=True)

    for i, f in enumerate(files, 1):
        filename = os.path.basename(f)
        out_path = os.path.join(output_dir, filename)

        ok = process_single_image(f, out_path, upsampler, scale)
        if ok:
            print(f"[{i}/{len(files)}] Done → {filename}")
        else:
            print(f"[{i}/{len(files)}] Failed → {filename}")


def main():
    parser = argparse.ArgumentParser(description="Upsample an image or folder with Real-ESRGAN.")
    parser.add_argument("--in", dest="input", required=True, help="Input image OR folder")
    parser.add_argument("--out", dest="output", required=True, help="Output folder")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4], help="Upscale factor")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    scale = args.scale

    print(f"[+] Using scale x{scale}")
    print("[+] Building Real-ESRGAN upsampler...")
    upsampler = build_upsampler(scale)

    # Case 1: input is a single image
    if os.path.isfile(input_path):
        # If output path is folder, keep same filename
        if os.path.isdir(output_path):
            filename = os.path.basename(input_path)
            out_file = os.path.join(output_path, filename)
        else:
            out_file = output_path

        print("[+] Processing single image...")
        ok = process_single_image(input_path, out_file, upsampler, scale)
        if ok:
            print(f"[✓] Saved → {out_file}")
        else:
            print("[!] Failed")

    # Case 2: input is a directory
    elif os.path.isdir(input_path):
        print("[+] Processing folder...")
        process_folder(input_path, output_path, upsampler, scale)

    else:
        print("[!] Invalid input path.")


if __name__ == "__main__":
    
    main()
