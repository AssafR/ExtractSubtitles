import collections
import numpy as np
from sortedcontainers import SortedDict

from utils import ImageCluster, pre_process_images, append_left_if_doesnt_exist, CORRELATION_THRESHOLD_FOR_MERGE, \
    images_too_different_in_size, transform_ecc, CORRELATION_THRESHOLD_FOR_DISMISSAL, FRACTION_OF_TOO_FAR_TO_ELIMINATE


class RegistrationResult(object):
    def __init__(self, img1, img2):
        if img1 is float:
            cc = img1
            (self.cc, self.warp_matrix, self.warped) = (cc, None, None)
        if img1 is None or images_too_different_in_size(img1, img2):
            (self.cc, self.warp_matrix, self.warped) = (0.0, None, None)
        else:
            (self.cc, self.warp_matrix, self.warped) = transform_ecc(img1, img2)

    def update(self, cc, warp_matrix, warped):
        self.cc = cc
        self.warp_matrix = warp_matrix
        self.warped = warped


class ClusterManager():
    def __init__(self, images_sql):
        self.base_images_sql = images_sql
        self.no_images = len(images_sql)
        self.clusters = SortedDict()  # All the clusters
        self.distances = SortedDict()
        self.init_clusters_and_distances(self.base_images_sql)
        self.processing_queue = collections.deque(maxlen=self.no_images + 1)
        self.next_processing_queue = collections.deque(maxlen=self.no_images + 1)
        self.processing_queue.extendleft(self.clusters.keys())  # Init the processing queue with all images

    def init_clusters_and_distances(self, images_sql):
        image_singleton_cluster: ImageCluster
        # Distances is a (sorted) dictionary of dictionaries, with distances[i][j] is the distance between i and j
        # Should be symmetrical, i.e. distances[i][j] == distances[j][i]
        # Initialize
        for img_serial_no, img_sql in enumerate(images_sql):
            image_id = img_sql.image_id  # img_serial_no  #
            process_image = pre_process_images([img_sql.image])[0]
            image_singleton_cluster = ImageCluster(total=1, avg_cc=0.0, representative_id=image_id,
                                                   avg_img=process_image, source_images=[image_id])
            self.clusters[image_id] = image_singleton_cluster
            self.distances[image_id] = SortedDict()
            self.distances[image_id][image_id] = (1.0, None, None)

    def cluster_letters(self):
        current = self.processing_queue.pop()  # Initialize with first image from queue
        new_distances_calculated_in_loop = 0
        while len(self.processing_queue) > 0:  # and len(next_processing_queue) > 0:
            next_cluster = self.processing_queue.pop()
            if self.clusters[next_cluster] is None or self.should_eliminate_cluster(next_cluster, self.no_images):
                print(f'Eliminating {next_cluster}')
                self.clusters[
                    next_cluster] = None  # The cluster is popped and not re-inserted, so will not be queried again
                continue

            if self.clusters[current] is None or self.should_eliminate_cluster(current, self.no_images):
                print(f'Eliminating {current}')
                self.clusters[current] = None  # The cluster is popped and not re-inserted, so will not be queried again
                current = None

            if len(self.processing_queue) == 0:  # Queue is empty
                processing_queue = self.next_processing_queue
                self.next_processing_queue = collections.deque(maxlen=self.no_images + 1)  # Clear the queue
                current = next_cluster
                if new_distances_calculated_in_loop == 0:  # A stop condition - nothing changed in this iteration
                    break
                continue

            if current is None or self.clusters[current] is None:
                current = next_cluster
                append_left_if_doesnt_exist(self.processing_queue, next_cluster)
                continue
            if next_cluster is None or self.clusters[next_cluster] is None:
                continue
            if current == next_cluster:
                append_left_if_doesnt_exist(self.next_processing_queue, next_cluster)
                continue

            print(f'Processing {current} and {next_cluster}')
            cc, cached = self.get_cc(self.distances, current, next_cluster)  # dist can be nan
            if not cached:  # A new value was calculated
                new_distances_calculated_in_loop = new_distances_calculated_in_loop + 1
            if np.isnan(cc):
                self.distances[current][next_cluster] = (0.0, None, None)
                self.distances[next_cluster][current] = (0.0, None, None)
                append_left_if_doesnt_exist(self.next_processing_queue, next_cluster)
                continue
            elif cc > CORRELATION_THRESHOLD_FOR_MERGE:
                print(f'Merging {current} and {next_cluster}')
                merged_cluster = self.merge_clusters(self.clusters, self.distances, current, next_cluster)
                if merged_cluster is not None:
                    append_left_if_doesnt_exist(self.next_processing_queue, merged_cluster.representative_id)
                    print(f'  --  Merged {current} and {next_cluster}')
                    current = merged_cluster.representative_id
                    continue
                else:
                    append_left_if_doesnt_exist(self.next_processing_queue, next_cluster)
            else:
                append_left_if_doesnt_exist(self.next_processing_queue, next_cluster)

        # Finished the clustering, now need to find the biggest clusters
        final_clusters = {k: v for k, v in self.clusters.items() if v is not None}
        sorted_clusters_by_total = sorted(final_clusters.items(), key=lambda x: x[1].total, reverse=True)
        biggest_clusters = dict(sorted_clusters_by_total)
        self.clusters = biggest_clusters  # Now sorted by total!

        return self.base_images_sql, self.no_images

    def merge_distances(self, distances: SortedDict, cluster1: ImageCluster, cluster2: ImageCluster):
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
            if cc1 > cc2:
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

        distances.pop(cluster2.representative_id, 0.0)  # Remove the cluster that was merged
        distances[cluster1.representative_id] = dist_dict_result  # Add the new combined cluster
        # TODO: Update all the other clusters that point to cluster2 to point to cluster1 instead

        return dist_dict_result

    def merge_clusters(self, clusters, distances, base_cluster_no, second_cluster_no):
        base_cluster: ImageCluster = clusters[base_cluster_no]
        second_cluster: ImageCluster = clusters[second_cluster_no]

        if base_cluster.representative_id > second_cluster.representative_id:  # Base is always with smaller id
            second_cluster, base_cluster = base_cluster, second_cluster

        merged_cluster: ImageCluster = base_cluster + second_cluster
        if merged_cluster.avg_cc is not None:
            clusters[base_cluster.representative_id] = None
            clusters[second_cluster.representative_id] = None
            clusters[merged_cluster.representative_id] = merged_cluster

            self.merge_distances(distances, base_cluster, second_cluster)
            return merged_cluster

        return None

    def get_cc(self, distances_dict, row, column):
        if row == column:
            return 1.0, True
        cached = False
        possibly_cached_registration = distances_dict[row].get(column, RegistrationResult(None, None))
        if possibly_cached_registration.cc is not None: # Cached value
            # print(f'     Cache hit for {row},{column} is {cached_cc}')
            cached = True
            cc = possibly_cached_registration.cc
        else: # Not cached value
            result = RegistrationResult(self.clusters[row].avg_img, self.clusters[column].avg_img)
            distances_dict[row][column] = result
            distances_dict[column][row] = result
            cc = result.cc
        return cc, cached

    def should_eliminate_cluster(self, cluster_no, total_clusters):
        num_images_too_far = 0
        num_images_total = 0
        for c, registration_result in self.distances[cluster_no].items():
            if self.clusters[c] is not None:
                num_images_total = num_images_total + self.clusters[c].total
                if registration_result.cc < CORRELATION_THRESHOLD_FOR_DISMISSAL:
                    num_images_too_far = num_images_too_far + self.clusters[c].total
                    if num_images_too_far > FRACTION_OF_TOO_FAR_TO_ELIMINATE * total_clusters:
                        return True
        if (num_images_too_far > FRACTION_OF_TOO_FAR_TO_ELIMINATE * total_clusters and
                num_images_too_far > FRACTION_OF_TOO_FAR_TO_ELIMINATE * num_images_total):
            return True
        return False
