# Source: https://magamig.github.io/posts/accurate-image-alignment-and-registration-using-opencv/

import cv2
import numpy as np
import copy
import utils
from dataclasses import dataclass, field

MINIMUM_ACCEPTED_CC = 0.9

orb = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    scoreType=cv2.ORB_HARRIS_SCORE)


def weighted_average(img1, img2, weight1, weight2):
    all_weight = weight1 + weight2
    img_combined_float = weight1 * img1.astype(np.float64) + (all_weight - weight1) * img2.astype(np.float64)
    img_combined_int = (img_combined_float / all_weight).astype(np.uint8)
    return img_combined_int


class FeatureExtraction:
    def __init__(self, img):
        self.img = copy.copy(img)
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.kps, self.des = orb.detectAndCompute( \
            self.gray_img, None)
        self.img_kps = cv2.drawKeypoints( \
            self.img, self.kps, 0, \
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.matched_pts = []


@dataclass(order=True)
class AverageImageStat:
    total: int
    avg_cc: float
    avg_img: np.ndarray = field(compare=False)
    base_img: np.ndarray = field(compare=False)


LOWES_RATIO = 0.7
MIN_MATCHES = 2  # 20
index_params = dict(
    algorithm=6,  # FLANN_INDEX_LSH
    table_number=6,
    key_size=10,
    multi_probe_level=2)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(
    index_params,
    search_params)


def feature_matching(features0, features1):
    matches = []  # good matches as per Lowe's ratio test
    if (features0.des is not None and len(features0.des) > 2):
        print(f"features0.des is not None and len(features0.des) = {len(features0.des)} > 2")
        all_matches = flann.knnMatch( \
            features0.des, features1.des, k=2)
        try:
            for m, n in all_matches:
                if m.distance < LOWES_RATIO * n.distance:
                    print('appending match: ', m)
                    matches.append(m)
                else:
                    print(f'Not appending: Distance is {m.distance}')
        except ValueError:
            print(ValueError)
            pass
        sorted_matches = sorted(matches, key=lambda x: x.distance, reverse=False)
        matches = sorted_matches  # [0:5]
        if (len(matches) > MIN_MATCHES):
            print("len(matches) > MIN_MATCHES)")
            features0.matched_pts = np.float32( \
                [features0.kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            features1.matched_pts = np.float32( \
                [features1.kps[m.trainIdx].pt for m in matches] \
                ).reshape(-1, 1, 2)
    print(features0.matched_pts, features1.matched_pts)
    return matches


def scale_convert_image(img: np.ndarray, scale_percent=200) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_factor = scale_percent / 100

    # resize image
    im_resized = cv2.resize(img_gray, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    return im_resized


@utils.CallCountDecorator
def transform_ECC(im1: np.ndarray, im2: np.ndarray) -> (float, np.ndarray, np.ndarray):
    # Source:  https://stackoverflow.com/questions/68497827/cv2-findtransformecc-how-to-ignore-small-particles

    convMode = "up"
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
        return 0.0, None, im2
    except AssertionError as e:
        return 0.0, None, im2


def calc_average_similar_base(base_index, images_enlarged):
    base_image = images_enlarged[base_index]
    sum_images = 0  #
    total = 0
    sum_cc = 0.0
    for img_no, img in enumerate(images_enlarged):
        if img_no == base_index:  # Optimize
            cc, warp_matrix, warped = 1.0, 1.0, base_image
        else:
            cc, warp_matrix, warped = transform_ECC(base_image, img)

        if warp_matrix is not None:
            # warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            sum_images = sum_images + warped.astype(np.float64)
            total = total + 1
            sum_cc = sum_cc + cc
        else:
            # disp(warped)
            pass
    if total == 0:
        return None
    avg_cc = sum_cc / total
    avg_img = (sum_images / total).astype(np.uint8)
    avg = AverageImageStat(total, avg_cc, avg_img, base_image)
    return avg


def warp_image_2(img1, img2):
    # https://stackoverflow.com/questions/55757977/how-to-use-estimaterigidtransform-in-opencv-3-0-or-higher-is-there-any-other-al
    # Use cv::estimateAffine2D, cv::estimateAffinePartial2D
    pass


def warp_image_1(img1, img2):
    features1 = FeatureExtraction(img1)
    features2 = FeatureExtraction(img2)
    matches = feature_matching(features1, features2)
    matched_image = cv2.drawMatches(img1, features1.kps, \
                                    img2, features2.kps, matches, None, flags=2)
    # disp(matched_image)
    h, w, c = img2.shape
    # H, _ = cv2.findHomography(features1.matched_pts, features2.matched_pts, cv2.RANSAC, 5.0)
    H, _ = cv2.estimateAffine2D(features1.matched_pts, features2.matched_pts)
    # H, _ = cv2.findTransformECC(img1,img2,)
    print(H)
    # warped = cv2.warpPerspective(img1, H, (w, h), \
    #                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped = cv2.warpAffine(img1, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped


def find_best_average_image_improved(images):
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
            aligned_images_matrix[row, column] = warped
            aligned_images_matrix[column, row] = warped

    # non_zeros = np.count_nonzero(distance_matrix, axis=0)
    non_zeros = np.count_nonzero(~np.isnan(distance_matrix), axis=1)
    good_rows = np.argwhere(non_zeros == non_zeros.max())
    good_rows = good_rows.reshape(len(good_rows))  # Convert to 1-D Vector
    avg_cc = (np.nanmean(distance_matrix[good_rows], axis=1))
    best_row = good_rows[np.argmax(avg_cc)]
    print(distance_matrix[best_row])
    print(non_zeros, best_row)
