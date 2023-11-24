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

sqlite_db = r'.\letters.sqlite'

DISTANCE_THRESHOLD_FOR_MERGE = 0.97

clusters = SortedDict()  # [] # All the clusters
distances = SortedDict()


@CallCountDecorator
def get_cc_cache(distance_matrix, row, column):
    return distance_matrix[row, column]

@CallCountDecorator
def get_cc(distance_matrix, row, column):
    cc, warp_matrix, warped = transform_ECC(clusters[row].avg_img, clusters[column].avg_img)
    return cc

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
        cc, warp_matrix, combined = create_augmented_image(self, other)
        new_avg_cc = cc
        new_avg_img = combined
        # Temporary solution, should be weighted average
        new_source_images = self.source_images + other.source_images

        return ImageCluster(total=new_total, avg_cc=new_avg_cc, representative_id=new_representative_id,
                            avg_img=new_avg_img, source_images=new_source_images)


def create_augmented_image(cluster1: ImageCluster, cluster2: ImageCluster):
    if cluster1.avg_img.shape[0] < cluster2.avg_img.shape[0]:
        cluster1, cluster2 = cluster1, cluster1  # 1 is the larger image
    cc, warp_matrix, warped = transform_ECC(cluster1.avg_img, cluster2.avg_img)
    combined = weighted_average(cluster1.avg_img, warped, cluster1.total, cluster2.total)
    return cc, warp_matrix, combined


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    letter = 'כ'  # 'ו'

    distance_matrix = read_create_distance_matrix(db, letter)
    images_sql = db.read_images_by_text_orderbyid(letter)

    cluster_singleton: tesseract_sql.ImageCluster
    # Distances is a (sorted) dictionary of dictionaries, with distances[i][j] is the distance between i and j
    # Initialize
    for img_serial_no, img_sql in enumerate(images_sql):
        image_id = img_serial_no  # img_sql.image_id
        process_image = tesseract_hebrew_utils.pre_process_images([img_sql.image])[0]
        cluster_singleton = ImageCluster(total=1, avg_cc=0.0, representative_id=image_id,
                                         avg_img=process_image, source_images=[image_id])
        # clusters.append(cluster_singleton)
        clusters[image_id] = cluster_singleton
        distances[image_id] = SortedDict()
        distances[image_id][image_id] = 1.0

    # Main loop
    merges = 1
    while merges > 0:
        current: int = clusters.keys()[-1]
        merges = 0
        print(f'Current cluster: {current}')
        for cluster_no in clusters.keys():
            if current == cluster_no:
                continue
            if clusters[cluster_no] is None or clusters[current] is None:
                continue

            cc = get_cc(distance_matrix, current, cluster_no)  # dist can be nan
            distances[cluster_no][current] = cc
            distances[current][cluster_no] = cc

            print(f'CC between {current} and {cluster_no} is {cc}')
            if pd.notna(cc):
                if cc > DISTANCE_THRESHOLD_FOR_MERGE:
                    print(f'  Merging clusters {current} and {cluster_no}')
                    merged_cluster = merge_clusters(clusters, distances, distance_matrix, current, cluster_no)
                    merges = merges + 1
                    current = merged_cluster.representative_id
                else:
                    print(f'  Should not merge clusters {current} and {cluster_no}')
            else:
                print(f'No cc between {current} and {cluster_no}')
        print(f'Loop finished with {merges} merges')

    print(f'Final clusters: ')
    for cluster_no in clusters.keys():
        if clusters[cluster_no]:
            print(f'Cluster #{cluster_no}: {clusters[cluster_no].total}')
        else:
            print(f'Cluster #{cluster_no}: None')
    # print(distance_matrix)
    # print(distance_matrix.shape)


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
            print(f'No distance for cluster {cluster_no}')
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
