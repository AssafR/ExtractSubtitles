import collections

import tesseract_hebrew_utils
import tesseract_sql
import numpy as np
import cv2
from sortedcontainers import SortedDict

from utils import (append_left_if_doesnt_exist, ImageCluster,
                   transform_ecc, get_cc, should_eliminate_cluster, disp, clusters, distances)

sqlite_db = r'.\letters2.sqlite'

LETTER = 'ו'

CORRELATION_THRESHOLD_FOR_MERGE = 0.94


# @CallCountDecorator
def get_cc_cache(distance_matrix, row, column):
    return distance_matrix[row, column]


def create_representative_letter(db, letter):
    images_sql = db.read_images_by_text_orderbyid(letter)
    no_images = len(images_sql)
    print(f'Number of images: {no_images}')
    if no_images == 0:
        return [], 0
    processing_queue = collections.deque(maxlen=no_images + 1)
    next_processing_queue = collections.deque(maxlen=no_images + 1)
    init_clusters_and_distances(images_sql)
    processing_queue.extendleft(clusters.keys())  # Init the processing queue with all images
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
            if new_distances_calculated_in_loop == 0:  # A stop condition - nothing changed in this iteration
                break
            continue

        if current is None or clusters[current] is None:
            current = next_cluster
            append_left_if_doesnt_exist(processing_queue, next_cluster)
            continue
        if next_cluster is None or clusters[next_cluster] is None:
            continue
        if current == next_cluster:
            append_left_if_doesnt_exist(next_processing_queue, next_cluster)
            continue

        print(f'Processing {current} and {next_cluster}')
        cc, cached = get_cc(distances, current, next_cluster)  # dist can be nan
        if not cached:  # A new value was calculated
            new_distances_calculated_in_loop = new_distances_calculated_in_loop + 1
        if np.isnan(cc):
            distances[current][next_cluster] = (0.0, None, None)
            distances[next_cluster][current] = (0.0, None, None)
            append_left_if_doesnt_exist(next_processing_queue, next_cluster)
            continue
        elif cc > CORRELATION_THRESHOLD_FOR_MERGE:
            print(f'Merging {current} and {next_cluster}')
            merged_cluster = merge_clusters(clusters, distances, current, next_cluster)
            if merged_cluster is not None:
                append_left_if_doesnt_exist(next_processing_queue, merged_cluster.representative_id)
                print(f'  --  Merged {current} and {next_cluster}')
                current = merged_cluster.representative_id
                continue
            else:
                append_left_if_doesnt_exist(next_processing_queue, next_cluster)
        else:
            append_left_if_doesnt_exist(next_processing_queue, next_cluster)
    return images_sql, no_images


def init_clusters_and_distances(images_sql):
    image_singleton_cluster: ImageCluster
    # Distances is a (sorted) dictionary of dictionaries, with distances[i][j] is the distance between i and j
    # Should be symmetrical, i.e. distances[i][j] == distances[j][i]
    # Initialize
    for img_serial_no, img_sql in enumerate(images_sql):
        image_id = img_sql.image_id  # img_serial_no  #
        process_image = tesseract_hebrew_utils.pre_process_images([img_sql.image])[0]
        image_singleton_cluster = ImageCluster(total=1, avg_cc=0.0, representative_id=image_id,
                                               avg_img=process_image, source_images=[image_id])
        clusters[image_id] = image_singleton_cluster
        distances[image_id] = SortedDict()
        distances[image_id][image_id] = (1.0, None, None)


def merge_distances(distances: SortedDict, cluster1: ImageCluster, cluster2: ImageCluster):
    # Two clusters are about to be merged together
    # Their distances should be updated accordingly.
    # Assumption: Cluster1 contains images i1,...,im but only i1 (smallest) is the representative id
    #             Cluster2 contains images j1,...,jn but only j1 (smallest) is the representative id
    # Cases:
    #  WLG, i1 is the new representative
    #    (i1,i1) = 1.0
    #    for each k<>i in Cluster1+Cluster2 : (i,k) and (k,i) should be deleted
    #    for each k<>i not in Cluster1/Cluster2: The new (i,k)|(k,i) should be the minimum
    #             of all (k,l) for l in (Cluster1+Cluster2 -> {i} join {j})

    # Assuming cluster1 has smaller representative id and so will remain representative of the combined cluster

    dist_dict_1: SortedDict = distances[cluster1.representative_id]
    dist_dict_2: SortedDict = distances[cluster2.representative_id]
    dist_dict_result = SortedDict()

    combined_clusters = set(dist_dict_1.keys()).union(set(dist_dict_2.keys()))
    for cluster_no in combined_clusters:  # Note: This will destroy the original dictionaries
        # Each cluster_no is a possible known distance to another cluster
        (cc1, warp_matrix1, warped1) = dist_dict_1.pop(cluster_no, (0.0, None, None))
        (cc2, warp_matrix2, warped2) = dist_dict_2.pop(cluster_no, (0.0, None, None))
        if (cc1 > cc2):
            (cc, warp_matrix, warped) = (cc1, warp_matrix1, warped1)
        else:
            (cc, warp_matrix, warped) = (cc2, warp_matrix2, warped2)

        dist_dict_result[cluster_no] = (cc, warp_matrix, warped)
        if cluster_no in distances:
            distances[cluster_no].pop(cluster2.representative_id, 0.0)
            distances[cluster_no][cluster1.representative_id] = (cc, warp_matrix, warped)
        else:
            # print(f'No distance for cluster {cluster_no}')
            pass
    # Now there's a new combined dictionary with the best correlation of the two groups

    distances.pop(cluster2.representative_id, 0.0)
    distances[cluster1.representative_id] = dist_dict_result

    return dist_dict_result


def merge_clusters(clusters, distances, base_cluster_no, second_cluster_no):
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


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    # letter = 'כ'  # 'ו'
    letter = LETTER

    images_sql, no_images = create_representative_letter(db, letter)

    display_letter_results(images_sql, no_images)


def display_letter_results(images_sql, no_images):
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
    non_aligned_images = tesseract_hebrew_utils.pre_process_images(non_aligned_images, 1.0)
    disp(tesseract_hebrew_utils.embed_images_in_square(non_aligned_images, 5))
    # print(distance_matrix)
    # print(distance_matrix.shape)


if __name__ == '__main__':
    main()
