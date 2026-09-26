import tesseract_hebrew_utils
import tesseract_sql
import cv2

import utils
from paths import CANONICAL_DATABASE_PATH

sqlite_db = CANONICAL_DATABASE_PATH


# view_image_wait_key(all_images)

# plt.imshow(all_images)
# plt.title('my picture')
# plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
# plt.show()

def disp(img):
    # display(Image.fromarray(all_images))
    utils.view_image_wait_key(img)


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db, create_if_missing=False)
    images_sql_unsorted = db.read_images_by_text_orderby_id('כ')
    images_sql = sorted(images_sql_unsorted, key=lambda x: x.image_id, reverse=False)
    images = [img.image for img in images_sql]

    images_enlarged = utils.pre_process_images(images[0:10], 2.0)

    all_images = tesseract_hebrew_utils.embed_images_in_square(images_enlarged, 6)
    disp(all_images)

    # plt.imshow(all_images)
    # plt.title('my picture')
    # plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
    # plt.show()

    # disp(all_images)

    tesseract_hebrew_utils.find_best_average_image_improved(images_enlarged)
    best_avg = tesseract_hebrew_utils.find_best_average_image(images_enlarged)

    # warped = warp_image_1(img1, img2)

    print(f'Used {best_avg.total} images of total {len(images_enlarged)}')
    avg = best_avg.avg_img

    # combined = ((img1.astype(int) + warped.astype(int)) / 2).astype(np.uint8)
    # disp(avg)
    _, avg_blur = cv2.threshold(avg, thresh=40, maxval=255, type=cv2.THRESH_BINARY)
    # avg_blur = cv2.medianBlur(avg_blur, 15)
    avg_blur = cv2.GaussianBlur(avg_blur, (9, 9), 10)

    # disp(avg_blur)

    disp(tesseract_hebrew_utils.hconcat_resize_max([avg, avg_blur]))

    avg_small = cv2.resize(avg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    print(avg_small.shape)
    avg_small_blur = cv2.resize(avg_blur, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    disp(tesseract_hebrew_utils.hconcat_resize_max([avg_small, avg_small_blur]))


if __name__ == '__main__':
    main()
