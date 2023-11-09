import tesseract_hebrew_utils
import tesseract_sql
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import copy
from alignment import FeatureExtraction, feature_matching

sqlite_db = r'.\letters.sqlite'


# view_image_wait_key(all_images)

# plt.imshow(all_images)
# plt.title('my picture')
# plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
# plt.show()

def disp(all_images):
    # display(Image.fromarray(all_images))
    tesseract_hebrew_utils.view_image_wait_key(all_images)


def process_images(images):
    images_enlarged = [cv2.copyMakeBorder(img, 10, 10, 10, 10,
                                          cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
                       for img in images]
    images_enlarged = [cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
                       for img in images_enlarged]
    return images_enlarged


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    images_sql = db.read_images_by_text('ל')
    images = [img.image for img in images_sql]

    images_enlarged = process_images(images)

    all_images = tesseract_hebrew_utils.embed_images_in_square(images_enlarged, 6)
    # disp(all_images)

    # plt.imshow(all_images)
    # plt.title('my picture')
    # plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
    # plt.show()

    # disp(all_images)

    img1, img2 = images_enlarged[0], images_enlarged[1]
    sum = np.zeros(img1.shape, np.float64)
    for img in images_enlarged:
        warped = cv2.cvtColor(transform_ECC(img1, img), cv2.COLOR_GRAY2BGR)
        sum = sum + warped.astype(np.float64)

    # warped = warp_image_1(img1, img2)

    avg = (sum / len(images_enlarged)).astype(np.uint8)

    # combined = ((img1.astype(int) + warped.astype(int)) / 2).astype(np.uint8)
    disp(avg)


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
    disp(matched_image)
    h, w, c = img2.shape
    # H, _ = cv2.findHomography(features1.matched_pts, features2.matched_pts, cv2.RANSAC, 5.0)
    H, _ = cv2.estimateAffine2D(features1.matched_pts, features2.matched_pts)
    # H, _ = cv2.findTransformECC(img1,img2,)
    print(H)
    # warped = cv2.warpPerspective(img1, H, (w, h), \
    #                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped = cv2.warpAffine(img1, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped


def transform_ECC(im1: np.ndarray, im2: np.ndarray):
    # Source:  https://stackoverflow.com/questions/68497827/cv2-findtransformecc-how-to-ignore-small-particles
    # Convert images to grayscale

    scale_percent = 100
    convMode = "up"
    num_iterations = 1000
    corr_coeff = 1e-5  # 0.5

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
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    return im2_aligned


if __name__ == '__main__':
    main()
