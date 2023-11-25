import collections

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
from tesseract_hebrew_utils import view_image_wait_key, disp
import cv2

sqlite_db = r'.\letters.sqlite'

CORRELATION_THRESHOLD_FOR_MERGE = 0.90
CORRELATION_THRESHOLD_FOR_DISMISSAL = 0.3

clusters = SortedDict()  # All the clusters
distances = SortedDict()


# @CallCountDecorator
def get_cc_cache(distance_matrix, row, column):
    return distance_matrix[row, column]


# @CallCountDecorator
def get_cc(distance_matrix, row, column):
    if row == column:
        return 1.0, True
    cached = False
    cached_cc = distances[row].get(column, None)
    if cached_cc is not None:
        # print(f'     Cache hit for {row},{column} is {cached_cc}')
        cached = True
        cc = cached_cc
    else:
        cc, warp_matrix, warped = transform_ECC(clusters[row].avg_img, clusters[column].avg_img)
        distances[row][column] = cc
        distances[column][row] = cc
    return cc, cached


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
        new_representative_id = min(self.representative_id, other.representative_id)
        cc, warp_matrix, combined = create_augmented_image(self, other)  # Weighted average
        new_avg_cc = cc
        new_avg_img = combined
        new_source_images = self.source_images + other.source_images

        return ImageCluster(total=new_total, avg_cc=new_avg_cc, representative_id=new_representative_id,
                            avg_img=new_avg_img, source_images=new_source_images)


def create_augmented_image(cluster1: ImageCluster, cluster2: ImageCluster):
    if cluster1.avg_img.shape[0] < cluster2.avg_img.shape[0]:
        cluster1, cluster2 = cluster1, cluster1  # 1 is the larger image
    cc, warp_matrix, warped = transform_ECC(cluster1.avg_img, cluster2.avg_img)
    combined = weighted_average(cluster1.avg_img, warped, cluster1.total, cluster2.total)
    return cc, warp_matrix, combined


def should_eliminate_cluster(cluster_no, total_clusters):
    num_clusters_too_far = 0
    for c, dist in distances[cluster_no].items():
        if dist < CORRELATION_THRESHOLD_FOR_DISMISSAL and clusters[c] is not None:
            num_clusters_too_far = num_clusters_too_far + clusters[c].total
            if num_clusters_too_far > 0.5 * total_clusters:
                return True
    return False


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    # letter = 'כ'  # 'ו'
    letter = 'ו'  # ''

    distance_matrix = read_create_distance_matrix(db, letter)
    images_sql = db.read_images_by_text_orderbyid(letter)
    no_images = len(images_sql)
    processing_queue = collections.deque(maxlen=no_images + 1)
    next_processing_queue = collections.deque(maxlen=no_images + 1)

    init_clusters_and_distances(images_sql)
    processing_queue.extendleft(clusters.keys())

    current = processing_queue.pop()  # Initialize with first image from queue
    new_distances_calculated_in_loop = 0
    while len(processing_queue) > 0:  # and len(next_processing_queue) > 0:
        next_cluster = processing_queue.pop()
        if clusters[next_cluster] is None or should_eliminate_cluster(next_cluster, no_images):
            print(f'Eliminating {next_cluster}')
            clusters[next_cluster] = None  # The cluster is popped and not re-inserted, so will not be queried again
            continue

        if clusters[current] is None or should_eliminate_cluster(current, no_images):
            print(f'Eliminating {current}')
            clusters[current] = None  # The cluster is popped and not re-inserted, so will not be queried again
            current = None

        if len(processing_queue) == 0:  # Queue is empty
            processing_queue = next_processing_queue
            next_processing_queue = collections.deque(maxlen=no_images + 1)  # Clear the queue
            current = next_cluster
            if new_distances_calculated_in_loop == 0:
                break
            continue

        if current is None or clusters[current] is None:
            current = next_cluster
            processing_queue.appendleft(next_cluster)
            continue
        if next_cluster is None or clusters[next_cluster] is None:
            continue
        if current == next_cluster:
            next_processing_queue.appendleft(next_cluster)
            continue

        print(f'Processing {current} and {next_cluster}')
        cc, cached = get_cc(distance_matrix, current, next_cluster)  # dist can be nan
        if not cached:  # A new value was calculated
            new_distances_calculated_in_loop = new_distances_calculated_in_loop + 1
        if np.isnan(cc):
            distances[current][next_cluster] = 0.0
            distances[next_cluster][current] = 0.0
            next_processing_queue.appendleft(next_cluster)
            continue
        elif cc > CORRELATION_THRESHOLD_FOR_MERGE:
            print(f'Merging {current} and {next_cluster}')
            merged_cluster = merge_clusters(clusters, distances, distance_matrix, current, next_cluster)
            if merged_cluster is not None:
                next_processing_queue.appendleft(merged_cluster.representative_id)
                print(f'  --  Merged {current} and {next_cluster}')
                current = merged_cluster.representative_id
                continue
            else:
                next_processing_queue.appendleft(next_cluster)
        else:
            next_processing_queue.appendleft(next_cluster)

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
    biggest_clusters_images = [cluster.avg_img for cluster_id,cluster in biggest_clusters.items() if cluster.total>1]
    biggest_clusters_images.append(biggest_clusters_images[0]) # Handle the edge case of size 1
    cc, warp_matrix, im_aligned = transform_ECC(biggest_clusters_images[0], biggest_clusters_images[1])
    print(f'cc={cc}')
    disp(tesseract_hebrew_utils.hconcat_resize_max(biggest_clusters_images,interpolation=cv2.INTER_CUBIC))

    # print(distance_matrix)
    # print(distance_matrix.shape)


def init_clusters_and_distances(images_sql):
    cluster_singleton: ImageCluster
    # Distances is a (sorted) dictionary of dictionaries, with distances[i][j] is the distance between i and j
    # Should be symmetrical, i.e. distances[i][j] == distances[j][i]
    # Initialize
    for img_serial_no, img_sql in enumerate(images_sql):
        image_id = img_sql.image_id  # img_serial_no  #
        process_image = tesseract_hebrew_utils.pre_process_images([img_sql.image])[0]
        cluster_singleton = ImageCluster(total=1, avg_cc=0.0, representative_id=image_id,
                                         avg_img=process_image, source_images=[image_id])
        clusters[image_id] = cluster_singleton
        distances[image_id] = SortedDict()
        distances[image_id][image_id] = 1.0


def merge_distances(distances: SortedDict, cluster1: ImageCluster, cluster2: ImageCluster):
    # Two clusters are about to be merged together
    # Their distances should be updated accordingly.
    # Assumption: if
    # Assumption: Cluster1 contains images i1,...,im but only i1 (smallest) is the representative id
    #             Cluster2 contains images j1,...,jn but only j1 (smallest) is the representative id
    # Cases:
    #  WLG, i1 is the new representative
    #    (i1,i1) = 1.0
    #    for each k<>i in Cluster1+Cluster2 : (i,k) and (k,i) should be deleted
    #    for each k<>i not in Cluster1/Cluster2: The new (i,k)/(k,i) should be the minimum
    #             of all (k,l) for l in (Cluster1+Cluster2 -> {i} join {j})

    # Assuming cluster1 has smaller representative id and so will remain representative of the combined cluster

    dist_dict_1: SortedDict = distances[cluster1.representative_id]
    dist_dict_2: SortedDict = distances[cluster2.representative_id]
    dist_dict_result = SortedDict()

    combined_clusters = set(dist_dict_1.keys()).union(set(dist_dict_2.keys()))
    for cluster_no in combined_clusters:  # Note: This will destroy the original dictionaries
        # Each cluster_no is a possible known distance to another cluster
        cc1 = dist_dict_1.pop(cluster_no, 0.0)
        cc2 = dist_dict_2.pop(cluster_no, 0.0)
        cc = max(cc1, cc2)
        dist_dict_result[cluster_no] = cc
        if cluster_no in distances:
            distances[cluster_no].pop(cluster2.representative_id, 0.0)
            distances[cluster_no][cluster1.representative_id] = cc
        else:
            # print(f'No distance for cluster {cluster_no}')
            pass
    # Now there's a new combined dictionary with the best correlation of the two groups

    distances.pop(cluster2.representative_id, 0.0)
    distances[cluster1.representative_id] = dist_dict_result

    return dist_dict_result


def merge_clusters(clusters, distances, distance_matrix, base_cluster_no, second_cluster_no):
    base_cluster: ImageCluster = clusters[base_cluster_no]
    second_cluster: ImageCluster = clusters[second_cluster_no]

    if base_cluster.representative_id > second_cluster.representative_id:  # Base is always with smaller id
        second_cluster, base_cluster = base_cluster, second_cluster

    merged_cluster: ImageCluster = base_cluster + second_cluster
    if merged_cluster.avg_cc is not None:
        clusters[base_cluster.representative_id] = None
        clusters[second_cluster.representative_id] = None
        clusters[merged_cluster.representative_id] = merged_cluster

        merge_distances(distances, base_cluster, second_cluster)

        return merged_cluster
    return None


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
    images = tesseract_hebrew_utils.pre_process_images(images_raw, 3.0)
    return images


if __name__ == '__main__':
    main()
