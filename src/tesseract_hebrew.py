import tesseract_sql
from tesseract_hebrew_utils import *

# INSERT OR IGNORE INTO aspect_corrections(aspect) VALUES(4.0);

sqlite_db = r'resources\letters.sqlite'

COLOR_GREEN = (0, 255, 0)


@deprecated
def new_image_to_boxes(image, lang, output_type, config=None):  # .splitlines() #, output_type=pytesseract.Output.DICT
    return image_to_boxes_keep_same(image, lang, output_type)


##########

def main_old():
    database = tesseract_sql.DatabaseManager(sqlite_db)
    # aspect = tesseract_sql.AspectCorrection(2.0)
    aspect = database.insert_aspect_correction(tesseract_sql.AspectCorrection(2.0))
    # aspect = database.insert_aspect_correction(tesseract_sql.AspectCorrection(4.0))
    # aspect = database.insert_aspect_correction(tesseract_sql.AspectCorrection(3.0))
    print(aspect)


def main():
    db = tesseract_sql.DatabaseManager(sqlite_db)
    images = db.read_images_by_text_orderby_id('ל')

    # for img in images:
    #     cv2.rectangle(img.image, (0, 0), (img.image.shape[1], img.image.shape[0]), (0, 255, 0), 3)

    images_enlarged = [cv2.resize(img.image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
                       for img in images]
    all_images = embed_images_in_square(images_enlarged, 6)
    # view_image_wait_key(all_images)

    img1, img2 = images_enlarged[0], images_enlarged[1]

    # all_images = hconcat_resize_max([img.image for img in images])  # cv2.hconcat(images)
    # cv2.imshow('img', all_images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
