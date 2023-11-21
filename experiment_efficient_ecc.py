from pathlib import Path

import tesseract_hebrew_utils
import tesseract_sql
import numpy as np

from alignment import transform_ECC

sqlite_db = r'.\letters.sqlite'

@CallCountDecorator
def get_cc(distance_matrix, row, column):
    return distance_matrix[row, column] 


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    letter = 'כ'

    filename = f'distance_matrix_{letter}.npy'
    if (Path(filename).exists()):
        distance_matrix = np.load(filename)
        print(distance_matrix)
    else:
        images = read_images_for_letter(db, letter)

        distance_matrix = calc_distance_matrix(images)
        np.save(filename, distance_matrix)


    print(distance_matrix)
    print(distance_matrix.shape)


def calc_distance_matrix(images):
    n_images = len(images)
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
            cc, warp_matrix, warped = transform_ECC(img_row, img_column)
            if warp_matrix is None or cc <= 0.0:
                cc = np.nan  # Value to fill
                warped = None
            distance_matrix[row, column] = cc
            distance_matrix[column, row] = cc
    return distance_matrix


def read_images_for_letter(db, letter):
    images_sql_unsorted = db.read_images_by_text(letter)
    images_sql = sorted(images_sql_unsorted, key=lambda x: x.image_id, reverse=False)
    images_raw = [img.image for img in images_sql]
    images = tesseract_hebrew_utils.pre_process_images(images_raw, 2.0)
    return images


if __name__ == '__main__':
    main()
