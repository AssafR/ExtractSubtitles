import re
from pathlib import Path
import pytesseract
import os.path
import datetime
import math
import numpy as np

from pytesseract import Output, run_and_get_output
from subprocess import check_output
from alignment import calc_average_similar_base, transform_ECC

import cv2

TESSERACT_EXE = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
BORDER_SIZE = 10


def view_image_wait_key(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class OcrBoxResult:

    def __init__(self, img, row):
        self.img = img
        self.hImg, self.wImg, _ = self.img.shape

        ocr_box_data = row.split()
        self.detected_char, self.page = ocr_box_data[0], int(ocr_box_data[5])
        self.left, self.top, self.right, self.bottom = int(ocr_box_data[1]), int(ocr_box_data[2]), int(
            ocr_box_data[3]), int(ocr_box_data[4])
        self.calc_img_bottom = self.hImg - self.bottom
        self.calc_img_top = self.hImg - self.top

    def extract_box_from_image(self, enlarge_factor=1.0):
        width = (self.right - self.left)
        height = (self.bottom - self.top)
        center_x = (self.left + self.right) / 2
        center_y = (self.top + self.bottom) / 2
        new_width = width * enlarge_factor
        new_height = height * enlarge_factor

        # Hope it's okay, didn't test edge cases yet
        new_left = max(int(center_x - new_width / 2), 0)
        new_right = min(int(center_x + new_width / 2), self.img.shape[1])
        new_bottom = min(int(center_y + new_height / 2), self.img.shape[0])
        new_top = max(int(center_y - new_height / 2), 0)

        char_box_original = self.img[self.hImg - self.bottom:self.hImg - self.top, self.left:self.right].copy()
        char_box_enlarged = self.img[self.hImg - new_bottom:self.hImg - new_top, new_left:new_right].copy()

        # view_image_wait_key(char_box_enlarged)
        print('-----')

        return char_box_original, char_box_enlarged


def new_char_filename(original_filename, letters_location, description_row):
    path = Path(original_filename)
    b = description_row.split()
    hex_str = b[0].encode("utf-8").hex()
    b[0] = hex_str + 'h_'
    new_filename = '_'.join(b)
    new_filename_full = path.with_stem(new_filename + '__' + path.stem).with_suffix('.png')
    new_filename_full = Path(letters_location).joinpath(new_filename_full.name)
    return new_filename_full.as_posix()


def find_best_average_image(images_enlarged):
    # Find by going over all n^2 possibilities
    avg_images_all_bases = []
    for base_image_no in range(0, len(images_enlarged)):
        print(f'Comparing to image #{base_image_no}')
        avg_image_info = calc_average_similar_base(base_image_no, images_enlarged)
        if avg_image_info is None:
            print(f'No average image for base {base_image_no}')
            continue
        avg_images_all_bases.append(avg_image_info)
        print(f' base image: {base_image_no},  total={avg_image_info.total} , cc={avg_image_info.avg_cc}')
    best_avg = sorted(avg_images_all_bases)[-1]
    return best_avg


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


##########
# Source: https://stackoverflow.com/questions/54246492/pytesseract-difference-between-image-to-string-and-image-to-boxes
# Modification
def image_to_boxes_keep_same(
        image,
        lang=None,
        config='',
        nice=0,
        output_type=pytesseract.Output.STRING,
        timeout=0,
):
    """
    Returns string containing recognized characters and their box boundaries
    """
    config = f'{config.strip()} makebox'
    args = [image, 'box', lang, config, nice, timeout]

    return {
        Output.BYTES: lambda: run_and_get_output(*(args + [True])),
        Output.STRING: lambda: run_and_get_output(*args),
    }[output_type]()


#######################

def get_file_date(file_name: Path):
    if os.path.exists(file_name):
        creation_timestamp = os.path.getctime(file_name)
        creation_datetime = datetime.datetime.fromtimestamp(creation_timestamp)
        return creation_datetime
    else:
        return None


def get_file_attributes(file_name):
    file_path = Path(file_name)
    return str(file_path.name), str(file_path.parent), get_file_date(file_name)


def perform_ocr_commandline(jpgfile, txt_filename, tesseract_exe=TESSERACT_EXE):
    cmd = f'"{tesseract_exe}" -l heb "{jpgfile}" "{txt_filename}"'
    print(f'Running\n{cmd}\n\n')
    output = check_output(cmd, shell=True).decode()
    print(output)
    print('----------')


# new_w = 122+118+116+4*3 = 368   (new width)
# black_img = np.zeros((209,new_w,3),dtype=np.uint8)
# Syntax is: image[top:bottom,left:right,:]
# print(im_list[0].shape)  (207, 122, 3),  207 is height!
# black_img[0:207,4:126:] = im_list[0]
# black_img[0:209,130:248,:] = im_list[1] (209, 118, 3)
# view_image_wait_key(black_img)

def insert_image(base_image, small_image, y, x):
    base_image[y:y + small_image.shape[0], x:x + small_image.shape[1]] = small_image


def embed_images_in_square(im_list, spacing):
    # Assumption: Images are roughly the same size
    h_max = max(im.shape[0] for im in im_list)
    w_max = max(im.shape[1] for im in im_list)
    images_in_line = math.ceil(math.sqrt(len(im_list)))
    no_lines = math.ceil(len(im_list) / images_in_line)
    h_total = (h_max + spacing) * no_lines + spacing
    w_total = (w_max + spacing) * images_in_line + spacing

    output_img = np.zeros((h_total, w_total), dtype=np.uint8)  # Black
    output_img[:, :] = 255

    for image_no, image in enumerate(im_list):
        img_row, img_col = divmod(image_no, images_in_line)
        img_pos_y = spacing + img_row * (h_max + spacing)
        img_pos_x = spacing + img_col * (w_max + spacing)
        insert_image(output_img, image, img_pos_y, img_pos_x)

    return output_img


# Interpolation methods:
#   ("area", cv2.INTER_AREA),
#   ("nearest", cv2.INTER_NEAREST),
#   ("linear", cv2.INTER_LINEAR),
#   ("cubic", cv2.INTER_CUBIC),
#   ("lanczos4", cv2.INTER_LANCZOS4)]

def hconcat_resize_max(im_list, interpolation=cv2.INTER_CUBIC):
    h_max = max(im.shape[0] for im in im_list)  # Resize all to same height for horizontal concatenation
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


# def pad_width_of_same_height_images(images,color=(255,255,255)):
#     # Assuming: All are of same height
#     w_max = max(im.shape[0] for im in images)
#     padded_images=[]
#     for im in images:


def resize_images_in_square(im_list, interpolation=cv2.INTER_CUBIC):
    h_max = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max), interpolation=interpolation)
                      for im in im_list]
    no_images = len(im_list_resize)
    no_images_h = math.ceil(math.sqrt(no_images))
    images_lines = []
    for line_no in range(no_images_h):
        line_images = im_list_resize[line_no * no_images_h:(line_no + 1) * no_images_h]
        images_lines.append(cv2.hconcat(line_images))


def pre_process_images(images, enlarge_ratio=None):
    images_enlarged = [cv2.copyMakeBorder(  # Convert to Greyscale and add border
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
        cv2.BORDER_CONSTANT, None, value=255)
        for img in images]
    images_enlarged = [255 - img for img in images_enlarged]  # Convert to negative (White on Black)
    if enlarge_ratio is not None:
        images_enlarged = [cv2.resize(img, None, fx=enlarge_ratio, fy=enlarge_ratio,
                                      interpolation=cv2.INTER_CUBIC)
                           for img in images_enlarged]
    return images_enlarged


class SubtitleDataFromFile(object):
    def __init__(self, filename):
        regex_pattern = r"^(.+)_([0-9]{1})([0-9]{4})([0-9]{4})([0-9]{4})([0-9]{4})([0-9]{4})([0-9]{4})$"
        match = re.match(regex_pattern, filename)

        if match:
            self.pBaseName = match.group(1)
            self.ln = int(match.group(2))
            self.x_min = int(match.group(3))
            self.y_min = int(match.group(4))
            self.w = int(match.group(5))
            self.h = int(match.group(6))
            self.W = int(match.group(7))
            self.H = int(match.group(8))
