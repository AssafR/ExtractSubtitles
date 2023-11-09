# Source: https://magamig.github.io/posts/accurate-image-alignment-and-registration-using-opencv/

import cv2
import numpy as np
import copy

orb = cv2.ORB_create(
    nfeatures=500,
    scaleFactor=1.2,
    scoreType=cv2.ORB_HARRIS_SCORE)


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


def transform_ECC(im1: np.ndarray, im2: np.ndarray):
    # Source:  https://stackoverflow.com/questions/68497827/cv2-findtransformecc-how-to-ignore-small-particles

    scale_percent = 500
    convMode = "up"
    num_iterations = 1000
    corr_coeff = 1e-5  # 0.5

    # Convert images to grayscale
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # percent of original size
    width = int(im1.shape[1] * scale_percent / 100)
    height = int(im1.shape[0] * scale_percent / 100)
    dim1 = (width, height)

    # percent of original size
    width = int(im2.shape[1] * scale_percent / 100)
    height = int(im2.shape[0] * scale_percent / 100)
    dim2 = (width, height)

    # resize image
    im1 = cv2.resize(im1, dim1, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, dim2, interpolation=cv2.INTER_AREA)

    # Find size of image1
    sz = im1.shape

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
        print(f'cc={cc}')

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        return cc, warp_matrix, im2_aligned
    except:
        return 0.0, None, im2
