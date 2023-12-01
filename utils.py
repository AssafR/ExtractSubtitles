import collections
import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from sortedcontainers import SortedDict

ACCEPTABLE_EXTRA_DIFFERENCE_IN_DIMENSIONS = 0.3  # 30%
ACCEPTABLE_RATIO_OF_DIFFERENCE_IN_DIMENSIONS = 1.0 + ACCEPTABLE_EXTRA_DIFFERENCE_IN_DIMENSIONS
MINIMUM_ACCEPTED_CC = 0.9

clusters = SortedDict()  # All the clusters
distances = SortedDict()


class ClusterManager():
    def __init__(self, cluster):
        self.cluster = cluster


class CallCountDecorator:
    """
    A decorator that will count and print how many times the decorated function was called
    """

    def __init__(self, inline_func):
        self.call_count = 0
        self.inline_func = inline_func

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self._print_call_count()
        return self.inline_func(*args, **kwargs)

    def _print_call_count(self):
        print(f" * The method {self.inline_func.__name__} called {self.call_count} times")


@CallCountDecorator
def transform_ecc(im1: np.ndarray, im2: np.ndarray) -> (float, np.ndarray, np.ndarray):
    # Source:  https://stackoverflow.com/questions/68497827/cv2-findtransformecc-how-to-ignore-small-particles

    convMode = "down" # "up"
    num_iterations = 1000
    corr_coeff = 1e-5  # 0.5

    # Find size of image1
    img_size = im1.shape

    # Define the motion model
    if convMode != "down":
        warp_mode = cv2.MOTION_EUCLIDEAN
    else:
        warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = int(num_iterations);

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = float(corr_coeff)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)
        # print(f'cc={cc}')
        assert cc >= MINIMUM_ACCEPTED_CC, "Correlation too low"
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (img_size[1], img_size[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (img_size[1], img_size[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

        cc = 1.0 - abs(1.0 - cc)  # Special fix: Handle case where cc>1 , wrap back from 1
        return cc, warp_matrix, im2_aligned
    except cv2.error as e:
        return 0.0, None, im1
    except AssertionError as e:
        return 0.0, None, im1


def weighted_average(img1, img2, weight1, weight2):
    assert img1.shape == img2.shape, "Images must have the same shape"
    all_weight = weight1 + weight2
    img_combined_float = weight1 * img1.astype(np.float64) + (all_weight - weight1) * img2.astype(np.float64)
    img_combined_int = (img_combined_float / all_weight).astype(np.uint8)
    return img_combined_int


def images_too_different_in_size(img1, img2, ratio=ACCEPTABLE_RATIO_OF_DIFFERENCE_IN_DIMENSIONS):
    """Returns True if the images are too different in size, False otherwise"""
    ratio1 = img1.shape[0] / img2.shape[0]
    ratio2 = img1.shape[1] / img2.shape[1]
    max_ratio = max(ratio1, 1.0 / ratio1, ratio2, 1.0 / ratio2)
    return max_ratio > ratio


def append_left_if_doesnt_exist(de_queue: collections.deque, element):
    """ Append element to the left of the queue if it doesn't exist in the queue already"""
    if element not in de_queue:
        de_queue.appendleft(element)


@dataclass(order=True)  # , eq=False
class ImageCluster:
    total: int
    avg_cc: float
    representative_id: int  # The "representative_id" of the group is the smallest id of all the images in the group
    avg_img: np.ndarray = field(compare=False)
    source_images: list = field(compare=False)

    default_transform_func = transform_ecc

    def __add__(self, other):
        # Implement addition behavior
        new_total = self.total + other.total
        new_representative_id = min(self.representative_id, other.representative_id)
        cc, warp_matrix, combined = create_combined_image_for_clusters(self, other,
                                                                       self.default_transform_func)  # Weighted average
        new_avg_cc = cc  # Note: if cc is None, means the augmented image was not constructed
        new_avg_img = combined
        new_source_images = self.source_images + other.source_images

        return ImageCluster(total=new_total, avg_cc=new_avg_cc, representative_id=new_representative_id,
                            avg_img=new_avg_img, source_images=new_source_images)


def get_file_attributes(file_name):
    file_path = Path(file_name)
    return str(file_path.name), str(file_path.parent), get_file_date(file_name)


def create_combined_image_for_clusters(cluster1: ImageCluster, cluster2: ImageCluster, transform_func):
    if cluster1.avg_img.shape[0] < cluster2.avg_img.shape[0]:
        cluster1, cluster2 = cluster1, cluster1  # 1 is the larger image
    cc, warp_matrix, warped = transform_func(cluster1.avg_img,
                                             cluster2.avg_img)  # Project smaller onto larger image transform_ecc
    combined = weighted_average(cluster1.avg_img, warped, cluster1.total, cluster2.total)
    return cc, warp_matrix, combined


def get_file_date(file_name: Path):
    if os.path.exists(file_name):
        creation_timestamp = os.path.getctime(file_name)
        creation_datetime = datetime.datetime.fromtimestamp(creation_timestamp)
        return creation_datetime
    else:
        return None


def get_cc(distances_dict, row, column):
    if row == column:
        return 1.0, True
    cached = False
    cached_cc, _, _ = distances_dict[row].get(column, (None, None, None))
    if cached_cc is not None:
        # print(f'     Cache hit for {row},{column} is {cached_cc}')
        cached = True
        cc = cached_cc
    else:
        if images_too_different_in_size(clusters[row].avg_img, clusters[column].avg_img):
            (cc, warp_matrix, warped) = (0.0, None, None)
        else:
            cc, warp_matrix, warped = transform_ecc(clusters[row].avg_img, clusters[column].avg_img)
        distances_dict[row][column] = (cc, warp_matrix, warped)
        distances_dict[column][row] = (cc, warp_matrix, warped)
    return cc, cached


def should_eliminate_cluster(cluster_no, total_clusters):
    num_images_too_far = 0
    num_images_total = 0
    for c, (cc, warp_matrix, warped) in distances[cluster_no].items():
        if clusters[c] is not None:
            num_images_total = num_images_total + clusters[c].total
            if cc < CORRELATION_THRESHOLD_FOR_DISMISSAL:
                num_images_too_far = num_images_too_far + clusters[c].total
                if num_images_too_far > FRACTION_OF_TOO_FAR_TO_ELIMINATE * total_clusters:
                    return True
    if (num_images_too_far > FRACTION_OF_TOO_FAR_TO_ELIMINATE * total_clusters and
            num_images_too_far > FRACTION_OF_TOO_FAR_TO_ELIMINATE * num_images_total):
        return True
    return False


def disp(img):
    view_image_wait_key(img)


CORRELATION_THRESHOLD_FOR_DISMISSAL = 0.7
FRACTION_OF_TOO_FAR_TO_ELIMINATE = 0.5


def view_image_wait_key(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
