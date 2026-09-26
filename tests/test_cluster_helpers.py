import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tesseract_hebrew_utils import embed_images_in_square
from utils import ImageCluster, create_combined_image_for_clusters, transform_ecc


def test_create_combined_image_preserves_the_shorter_cluster():
    smaller_image = np.zeros((2, 4), dtype=np.uint8)
    larger_image = np.zeros((4, 4), dtype=np.uint8)
    smaller_cluster = ImageCluster(1, 0.0, 1, smaller_image, [1])
    larger_cluster = ImageCluster(2, 0.0, 2, larger_image, [2, 3])
    received_images = []

    def transform(template, target):
        received_images.extend((template, target))
        return 0.95, np.eye(3), np.zeros_like(template)

    _, _, combined = create_combined_image_for_clusters(
        smaller_cluster,
        larger_cluster,
        transform,
    )

    assert received_images[0] is larger_image
    assert received_images[1] is smaller_image
    assert combined.shape == larger_image.shape


def test_embed_images_in_square_accepts_text_label():
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    output = embed_images_in_square([image], spacing=1, text="Source Images")

    assert output.ndim == 3
    assert output.shape[2] == 3


@pytest.mark.parametrize(
    ("warp_mode", "warp_shape", "warp_function"),
    [
        (cv2.MOTION_EUCLIDEAN, (2, 3), "affine"),
        (cv2.MOTION_AFFINE, (2, 3), "affine"),
        (cv2.MOTION_HOMOGRAPHY, (3, 3), "perspective"),
    ],
)
def test_transform_ecc_uses_requested_warp_mode(
    monkeypatch, warp_mode, warp_shape, warp_function
):
    template = np.zeros((8, 8), dtype=np.uint8)
    input_image = np.ones((8, 8), dtype=np.uint8)
    calls = []

    def fake_find_transform_ecc(template_image, target_image, matrix, actual_mode, criteria):
        calls.append((actual_mode, matrix.shape))
        return 0.95, matrix

    def fake_warp_affine(image, matrix, size, flags):
        calls.append(("affine", size))
        return image

    def fake_warp_perspective(image, matrix, size, flags):
        calls.append(("perspective", size))
        return image

    monkeypatch.setattr(cv2, "findTransformECC", fake_find_transform_ecc)
    monkeypatch.setattr(cv2, "warpAffine", fake_warp_affine)
    monkeypatch.setattr(cv2, "warpPerspective", fake_warp_perspective)

    _, matrix, aligned = transform_ecc(template, input_image, warp_mode)

    assert calls[0] == (warp_mode, warp_shape)
    assert calls[1][0] == warp_function
    assert matrix.shape == warp_shape
    assert aligned is input_image
