import collections

import tesseract_hebrew_utils
import tesseract_sql
import numpy as np
import cv2
from sortedcontainers import SortedDict
import utils
from utils import (ImageCluster, disp, transform_ecc)
from cluster import ClusterManager

sqlite_db = r'.\letters2.sqlite'

LETTER = 'ל'


# @CallCountDecorator
def get_cc_cache(distance_matrix, row, column):
    return distance_matrix[row, column]


def create_representative_letter(db, letter):
    images_sql = db.read_images_by_text_orderbyid(letter)
    no_images = len(images_sql)
    print(f'Number of images: {no_images}')
    if no_images == 0:
        return [], 0

    cluster_manager = ClusterManager(images_sql)
    cluster_manager.cluster_letters()
    return images_sql, no_images, cluster_manager


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    # letter = 'כ'  # 'ו'
    letter = LETTER

    images_sql, no_images, cluster_manager = create_representative_letter(db, letter)

    display_letter_results(images_sql, no_images, cluster_manager.clusters)


def display_letter_results(images_sql, no_images, clusters):
    print(f'Final clusters: ')
    final_clusters = {k: v for k, v in clusters.items() if v is not None}
    sorted_clusters_by_total = sorted(final_clusters.items(), key=lambda x: x[1].total, reverse=True)
    biggest_clusters = dict(sorted_clusters_by_total)
    for cluster_no in biggest_clusters.keys():
        if clusters[cluster_no]:
            print(f'Cluster #{cluster_no}: {clusters[cluster_no].total}')
        # else:
        #     print(f'Cluster #{cluster_no}: None')
    print(f'Total images: {no_images}')
    biggest_clusters_images = [cluster.avg_img for cluster_id, cluster in biggest_clusters.items() if cluster.total > 1]
    biggest_clusters_images.append(biggest_clusters_images[0])  # Handle the edge case of size 1
    cc, warp_matrix, im_aligned = transform_ecc(biggest_clusters_images[0], biggest_clusters_images[1])
    print(f'cc={cc}')
    disp(tesseract_hebrew_utils.hconcat_resize_max(biggest_clusters_images, interpolation=cv2.INTER_CUBIC))
    best_cluster: ImageCluster = clusters[list(biggest_clusters.keys())[0]]
    non_aligned_images = [image_sql.image for image_sql in images_sql if
                          image_sql.image_id not in best_cluster.source_images]
    non_aligned_images = utils.pre_process_images(non_aligned_images, 1.0)
    disp(tesseract_hebrew_utils.embed_images_in_square(non_aligned_images, 5))
    # print(distance_matrix)
    # print(distance_matrix.shape)


if __name__ == '__main__':
    main()
