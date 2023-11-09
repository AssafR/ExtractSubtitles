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
    images_enlarged = [cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)
                       for img in images_enlarged]
    return images_enlarged


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    images_sql = db.read_images_by_text('ל')
    images = [img.image for img in images_sql]

    images_enlarged = process_images(images)

    all_images = tesseract_hebrew_utils.embed_images_in_square(images_enlarged, 6)
    disp(all_images)

    # plt.imshow(all_images)
    # plt.title('my picture')
    # plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
    # plt.show()

    # disp(all_images)

    img1, img2 = images_enlarged[0], images_enlarged[1]
    warped = warp_image_1(img1, img2)
    disp(warped)


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
    warped = cv2.warpAffine(img1,H,(w,h),borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped


if __name__ == '__main__':
    main()
