import tesseract_hebrew_utils
import tesseract_sql
from alignment import transform_ECC, weighted_average
from utils import CallCountDecorator
from dataclasses import dataclass, field
from pathlib import Path
from sortedcontainers import SortedDict
import numpy as np
import pandas as pd
from collections import OrderedDict

sqlite_db = r'.\letters.sqlite'

DISTANCE_THRESHOLD_FOR_MERGE = 0.97


@CallCountDecorator
def get_cc(distance_matrix, row, column):
    return distance_matrix[row, column]


@dataclass(order=True)  # , eq=False
class ImageCluster:
    total: int
    avg_cc: float
    representative_id: int
    avg_img: np.ndarray = field(compare=False)
    source_images: list = field(compare=False)

    def __add__(self, other):
        # Implement addition behavior
        new_total = self.total + other.total
        new_avg_cc = (self.avg_cc + other.avg_cc) / 2  # Averaging for simplicity
        new_representative_id = min(self.representative_id, other.representative_id)
        new_avg_img = weighted_average(self.avg_img, other.avg_img, self.total, other.total)
        new_source_images = self.source_images + other.source_images

        return ImageCluster(total=new_total, avg_cc=new_avg_cc, representative_id=new_representative_id,
                            avg_img=new_avg_img, source_images=new_source_images)


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    letter = 'כ'  # 'ו'

    distance_matrix = read_create_distance_matrix(db, letter)
    images_sql = db.read_images_by_text_orderbyid(letter)

    clusters = SortedDict()  # [] # All the clusters
    cluster_singleton: tesseract_sql.ImageCluster
    distances = SortedDict()
    for img_serial_no, img_sql in enumerate(images_sql):
        image_id = img_serial_no  # img_sql.image_id
        cluster_singleton = ImageCluster(total=1, avg_cc=0.0, representative_id=image_id,
                                         avg_img=img_sql.image, source_images=[image_id])
        # clusters.append(cluster_singleton)
        clusters[image_id] = cluster_singleton
        distances[image_id] = SortedDict()
        distances[image_id][image_id] = 1.0

    current: int = clusters.keys()[1]
    for cluster_no in clusters.keys():
        if current == cluster_no:
            continue
        if clusters[cluster_no] is None or clusters[current] is None:
            continue

        cc = get_cc(distance_matrix, current, cluster_no)  # dist can be nan
        print(f'CC between {current} and {cluster_no} is {cc}')
        if pd.notna(cc):
            if cc > DISTANCE_THRESHOLD_FOR_MERGE:
                print(f'  Should merge clusters {current} and {cluster_no}')
                merge_clusters(clusters, distances, distance_matrix, current, cluster_no)
            else:
                print(f'  Should not merge clusters {current} and {cluster_no}')
        else:
            print(f'No cc between {current} and {cluster_no}')

    # print(distance_matrix)
    # print(distance_matrix.shape)


def merge_clusters(clusters, distances, distance_matrix, base_cluster_no, merged_cluster_no):
    pass


def read_create_distance_matrix(db, letter):
    filename = f'distance_matrix_{letter}.npy'
    if Path(filename).exists():
        distance_matrix = np.load(filename)
        print(distance_matrix)
    else:
        images = read_images_for_letter(db, letter)

        distance_matrix = calc_distance_matrix(images)
        np.save(filename, distance_matrix)
    return distance_matrix


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
    images_sql = db.read_images_by_text_orderbyid(letter)
    # images_sql = sorted(images_sql, key=lambda x: x.image_id, reverse=False)
    images_raw = [img.image for img in images_sql]
    images = tesseract_hebrew_utils.pre_process_images(images_raw, 2.0)
    return images


if __name__ == '__main__':
    main()
