from pathlib import Path

from src.utils import transform_ecc
from src.tesseract_sql import read_images_for_letter
import numpy as np

def read_create_distance_matrix(db, letter):
    filename = f'distance_matrix_{letter}.npy'
    if Path(filename).exists():
        distance_matrix = np.load(filename)
        # print(distance_matrix)
    else:
        images = read_images_for_letter(db, letter)

        distance_matrix = calc_distance_matrix(images)
        np.save(filename, distance_matrix)
    return distance_matrix


def calc_distance_matrix(images):
    n_images: int = len(images)
    distance_matrix = np.zeros((n_images, n_images),
                               dtype=np.float32)  # np.full((n_images, n_images), np.nan, dtype=np.float32)
    aligned_images_matrix = np.full((n_images, n_images), None,
                                    dtype=np.ndarray)
    np.fill_diagonal(distance_matrix, 1.0)
    np.fill_diagonal(aligned_images_matrix, images)
    for (row, img_row) in enumerate(images):
        for (column, img_column) in enumerate(images):
            if row >= column:  # Fill only half the matrix
                continue
            if distance_matrix[row, column] > 0.0 or np.isnan(distance_matrix[row, column]):  # Already calculated
                continue
            cc, warp_matrix, warped = transform_ecc(img_row, img_column)
            if warp_matrix is None or cc <= 0.0:
                cc = np.nan  # Value to fill
                warped = None
            distance_matrix[row, column] = cc
            distance_matrix[column, row] = cc
    return distance_matrix
