import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from rank_pooling import rank_pooling
from video_io import read_frames

parser = argparse.ArgumentParser(
    description="Generate dynamic image",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("video_folder", type=Path)
parser.add_argument("dynamic_img", type=Path)
parser.add_argument(
    "-n", "--non-linearity", default="ssr", choices=["ssr", "ref", "tanh", "none"]
)


def main(argv=None):
    args = parser.parse_args(argv)
    video_frames = read_frames(args.video_folder).astype(np.float32)
    non_linearity = args.non_linearity
    dynamic_image = create_dynamic_image(video_frames, non_linearity)
    Image.fromarray(dynamic_image).save(str(args.dynamic_img))


def create_dynamic_image(video_frames, non_linearity="ssr"):
    t, h, w, c = video_frames.shape
    dynamic_image = rank_pooling(
        video_frames.reshape(t, -1).T, non_linearity=non_linearity
    ).T.reshape(h, w, c)
    vmin = dynamic_image.min()
    vmax = dynamic_image.max()
    dynamic_image = np.clip(
        255 * (dynamic_image - vmin) / (vmax - vmin), 0, 255
    ).astype(np.uint8)
    return dynamic_image


if __name__ == "__main__":
    main()
