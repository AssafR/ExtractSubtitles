import tesseract_hebrew_utils
import tesseract_sql
import cv2
import utils
from utils import (ImageCluster, disp, transform_ecc)
from cluster import ClusterManager
from paths import CANONICAL_DATABASE_PATH

sqlite_db = CANONICAL_DATABASE_PATH

LETTER = 'ל'


# @CallCountDecorator
def get_cc_cache(distance_matrix, row, column):
    return distance_matrix[row, column]


def create_representative_letter(db, letter):
    images_sql = db.read_images_by_text_orderby_id(letter)
    no_images = len(images_sql)
    print(f'Number of images: {no_images}')
    if no_images == 0:
        return [], 0, None

    cluster_manager = ClusterManager(images_sql)
    cluster_manager.cluster_letters()
    return images_sql, no_images, cluster_manager


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db, create_if_missing=False)
    # letter = 'כ'  # 'ו'
    letter = LETTER

    images_sql, no_images, cluster_manager = create_representative_letter(db, letter)
    if cluster_manager is None:
        print(f'No images found for letter: {letter}')
        return

    display_letter_results(images_sql, no_images, cluster_manager.clusters, cluster_manager.avg_image)


def display_letter_results(images_sql, no_images, largest_clusters, image):
    print(f'Final clusters: ')
    for cluster_no in largest_clusters.keys():
        if largest_clusters[cluster_no]:
            print(f'Cluster #{cluster_no}: {largest_clusters[cluster_no].total}')
        # else:
        #     print(f'Cluster #{cluster_no}: None')
    print(f'Total images: {no_images}')
    surviving_clusters = [cluster for cluster in largest_clusters.values() if cluster is not None]
    if not surviving_clusters:
        print('No surviving clusters to display')
        return

    biggest_clusters_images = [cluster.avg_img for cluster in surviving_clusters if cluster.total > 1]
    if not biggest_clusters_images:
        biggest_clusters_images = [surviving_clusters[0].avg_img]
    comparison_images = biggest_clusters_images if len(biggest_clusters_images) > 1 else biggest_clusters_images * 2
    cc, warp_matrix, im_aligned = transform_ecc(comparison_images[0], comparison_images[1])
    print(f'cc={cc}')
    disp(image, 'avg_image')
    disp(tesseract_hebrew_utils.hconcat_resize_max(comparison_images + [image], interpolation=cv2.INTER_CUBIC), 'biggest_clusters_images')
    best_cluster: ImageCluster = surviving_clusters[0]
    non_aligned_images = [image_sql.image for image_sql in images_sql if
                          image_sql.image_id not in best_cluster.source_images]
    if non_aligned_images:
        non_aligned_images = utils.pre_process_images(non_aligned_images, 1.0)
        disp(tesseract_hebrew_utils.embed_images_in_square(non_aligned_images, 5), 'Non aligned images')
    # print(distance_matrix)
    # print(distance_matrix.shape)


if __name__ == '__main__':
    main()
