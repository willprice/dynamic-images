from pathlib import Path

import numpy as np
from PIL import Image


def read_frames(
    video_folder: Path, img_extensions=(".png", ".jpg", ".jpeg")
) -> np.ndarray:
    frame_paths = sorted(
        [f for f in video_folder.iterdir() if f.suffix.lower() in img_extensions]
    )
    return np.stack([np.asarray(Image.open(path)) for path in frame_paths])