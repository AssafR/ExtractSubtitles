import glob
import sys
import cv2
import pytesseract
from pytesseract import Output, run_and_get_output
from pathlib import Path
import tesseract_sql
from tesseract_hebrew_utils import *

# INSERT OR IGNORE INTO aspect_corrections(aspect) VALUES(4.0);

sqlite_db = r'.\letters.sqlite'

COLOR_GREEN = (0, 255, 0)
ASPECT_RATIO_CORRECTION = 2.0
TESSERACT_CUSTOM_CONFIG_STR = r'--oem 3 --psm 6 -l heb'


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
    print('|'.join(sys.argv))
    jpeg_location = sys.argv[1]
    txt_location = sys.argv[2]
    letters_location = sys.argv[3]

    print(f'Jpeg Location: {jpeg_location}')
    txt_path = Path(txt_location)
    # args is a list of the command line args
    jpgfiles = glob.glob(jpeg_location + '/*.jpeg')
    print(len(jpgfiles))
    print(jpgfiles)

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

    db = tesseract_sql.DatabaseManager(sqlite_db)

    # Save the (currently global) aspect ratio
    aspect = db.insert_aspect_correction(tesseract_sql.AspectCorrection(ASPECT_RATIO_CORRECTION))  # Currently constant

    create_db_from_jpgfiles(aspect, db, jpgfiles, letters_location, txt_path)

    images = db.read_images_by_text_orderbyid('ל')

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


def create_db_from_jpgfiles(aspect, db, jpgfiles, letters_location, txt_path):
    for jpgfile in jpgfiles:
        filename = Path(jpgfile).stem
        txt_filename = txt_path.joinpath(filename)
        # perform_ocr_commandline(jpgfile, txt_filename)
        perform_ocr_api_save_to_db(db, aspect, jpgfile, txt_filename, letters_location)


def perform_ocr_api_save_to_db(db: tesseract_sql.DatabaseManager,
                               aspect: tesseract_sql.AspectCorrection,
                               jpgfile: str, txt_filename, letters_location):
    # Currently unused. Note pBaseName should be parsed too, e.g: '0_40_23_280__0_40_27_479'
    # subtitle_data = SubtitleDataFromFile(Path(jpgfile).stem)
    # print(subtitle_data)

    sub_file = db.insert_subs_files(tesseract_sql.SubsFiles(jpgfile))  # Save the filename to database

    full_img = cv2.imread(sub_file.full_file_name())

    char_boxes = {}

    hImg, wImg, _ = full_img.shape
    full_img = cv2.resize(full_img, (int(aspect.aspect * wImg), int(hImg)))
    hImg, wImg, _ = full_img.shape

    ratio = 1024.0 / wImg;

    # img = cv2.resize(img, (int(ratio * wImg), int(ratio * hImg)))

    # Adding custom options
    detected_text = pytesseract.image_to_string(full_img, config=TESSERACT_CUSTOM_CONFIG_STR)
    # d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(detected_text)
    print('----------')

    string_boxes = pytesseract.image_to_boxes(full_img, lang="heb",
                                              output_type=pytesseract.Output.STRING).splitlines()  # , output_type=pytesseract.Output.DICT
    # string_boxes = pytesseract.image_to_data(img, lang="heb",
    #                                          output_type=pytesseract.Output.DICT)  # .splitlines() #, output_type=pytesseract.Output.DICT
    ocr_box_results = [OcrBoxResult(full_img, row) for row in string_boxes]
    for ocr_result in ocr_box_results:
        # detected_box_file_name = new_char_filename(jpgfile, letters_location, row)
        # detected_char, left, top, right, bottom = \
        #     string_boxes['char'][index],string_boxes['left'][index],string_boxes['top'][index],string_boxes['right'][index],string_boxes['bottom'][index]
        # detected_word, left, top, width, height = \
        #     string_boxes['text'][index], string_boxes['left'][index], string_boxes['top'][index], string_boxes['width'][
        #         index], string_boxes['height'][index]
        # (x1,y1), (x2,y2)
        # cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (255, 0, 0), 2)
        # print(b[1:])  # Row of numbers

        print(f"\n** {ocr_result.detected_char} **")

        tmp_img = full_img.copy()

        char_box, char_box_enlarged = ocr_result.extract_box_from_image(enlarge_factor=1.2)
        img_data = tesseract_sql.Image(id=None, text=ocr_result.detected_char, img=char_box, hash=None)
        # Store image in dataclass
        if char_box.size > 0:
            # cursor = db.conn.cursor()
            # cursor.execute("INSERT INTO images (image) values (?)", (char_box, ))
            # db.conn.commit()
            # inserted_id = cursor.lastrowid
            img_data = db.insert_image(img_data)
            # print(img_data)
        else:
            print('Box is empty!')

        subs_result = tesseract_sql.SubsDecoded(
            subs_decoded_id=None,
            sub_file_fk=sub_file.sub_file_id,
            image_id_fk=img_data.image_id,
            detected_char=img_data.image_text,
            left=ocr_result.left,
            right=ocr_result.right,
            top=ocr_result.top,
            bottom=ocr_result.bottom,
            page=ocr_result.page,
        )
        subs_result = db.insert_subs_decoded(subs_result)
        print(str(subs_result))

        # sql_img = tesseract_sql.Image(image_id=None, image=char_box)
        # sql_img = db.insert_image(sql_img)
        # print(sql_img)

        # cv2.rectangle(tmp_img, (left, calc_img_top), (right, calc_img_bottom), COLOR_GREEN, 3)

        # Path(detected_box_file_name).parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(detected_box_file_name, char_box)
        # char_boxes[detected_char] = char_boxes.get(detected_char, []) + [char_box_enlarged]

        # img_resized = cv2.resize(tmp_img, (int(ratio * wImg), int(ratio * hImg)))

        # cv2.imshow('img', char_box)
        # cv2.waitKey(0)
        # print('-----')

        # if detected_word:
        #     word_rect_img = img[top - 2:top + height + 4, left - 2:left + width + 4]
        #     # char_boxes = pytesseract.image_to_boxes(word_rect_img, lang="heb",
        #     #                                         output_type=pytesseract.Output.DICT)  # .splitlines() #, output_type=pytesseract.Output.DICT
        #     char_boxes = new_image_to_boxes(word_rect_img, lang="heb",
        #                                     output_type=pytesseract.Output.STRING)  # .splitlines() #, output_type=pytesseract.Output.DICT
        #
        #     # cv2.imshow('img', word_rect_img)
        #     # cv2.waitKey(0)
        #     for index_char in range(len(char_boxes['left'])):
        #         detected_char, left, top, right, bottom = \
        #             string_boxes['char'][index], string_boxes['left'][index], string_boxes['top'][index], \
        #                 string_boxes['right'][index], string_boxes['bottom'][index]
        #         # cv2.putText(img, str(detected_word), (left, top + 13), cv2.QT_FONT_BLACK, 0.4, (50, 205, 50), 1)
        #         cv2.rectangle(img, (left, top), (right, bottom), (50, 0, 50), 2)

    # for char, images in char_boxes.items():
    #     print(f'*** {char} ***')
    #     ocr_from_boxes = ''
    #     for img in images:
    #         re_ocr = pytesseract.image_to_string(img, lang='heb').strip()
    #         ocr_from_boxes += re_ocr
    #         cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
    #
    #     print(f"Re-interpreted: [{'|'.join(ocr_from_boxes)}]")
    #     # all_images = hconcat_resize_max(images)  # cv2.hconcat(images)
    #     # cv2.imshow('img', all_images)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #
    # img_resized = cv2.resize(full_img, (int(ratio * wImg), int(ratio * hImg)))
    # cv2.imshow('img', img_resized)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    return None


if __name__ == '__main__':
    main()
