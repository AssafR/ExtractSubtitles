import tesseract_hebrew_utils
import tesseract_sql
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import copy
from alignment import FeatureExtraction, feature_matching, transform_ECC

sqlite_db = r'.\letters.sqlite'

BORDER_SIZE = 10

# view_image_wait_key(all_images)

# plt.imshow(all_images)
# plt.title('my picture')
# plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
# plt.show()

def disp(all_images):
    # display(Image.fromarray(all_images))
    tesseract_hebrew_utils.view_image_wait_key(all_images)


def process_images(images):
    images_enlarged = [cv2.copyMakeBorder(img, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                          cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
                       for img in images]
    images_enlarged = [255 - img  for img in images_enlarged] # Convert to negative (White on Black)
    images_enlarged = [cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
                       for img in images_enlarged]
    return images_enlarged


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    images_sql = db.read_images_by_text('כ')
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
    sum_images = 0 #
    # np.zeros(img1.shape, np.float64)
    total = 0
    for img in images_enlarged:
        cc, warp_matrix, warped = transform_ECC(img1, img)
        if warp_matrix is not None:
            warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            sum_images = sum_images + warped.astype(np.float64)
            total = total + 1
        else:
            disp(warped)

    # warped = warp_image_1(img1, img2)

    avg = (sum_images / total).astype(np.uint8)
    print(f'Used {total} images of total {len(images_enlarged)}')

    # combined = ((img1.astype(int) + warped.astype(int)) / 2).astype(np.uint8)
    disp(avg)
    disp(cv2.resize(avg, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC))


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




if __name__ == '__main__':
    main()
