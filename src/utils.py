import collections
import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import cv2
import numpy
import numpy as np

ACCEPTABLE_EXTRA_DIFFERENCE_IN_DIMENSIONS = 0.3  # 30%
ACCEPTABLE_RATIO_OF_DIFFERENCE_IN_DIMENSIONS = 1.0 + ACCEPTABLE_EXTRA_DIFFERENCE_IN_DIMENSIONS
MINIMUM_ACCEPTED_CC = 0.9
BORDER_SIZE = 10

CORRELATION_THRESHOLD_FOR_MERGE = 0.94
CORRELATION_THRESHOLD_FOR_DISMISSAL = 0.7
FRACTION_OF_TOO_FAR_TO_ELIMINATE = 0.5

def disp(img, title=None):
    """Show an image and wait until the display window is closed."""
    view_image_wait_key(img, title)


def view_image_wait_key(img, title=None):
    """Display an image with OpenCV and wait for a key press."""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trim_to_smallest_rectangle(original_image):
    """Crop an image to the first external foreground contour."""
    # Read the image
    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(original_image, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no contours found, return the original image
        return original_image

    # Find the bounding box of the contours
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image to the bounding box
    trimmed_image = original_image[y:y + h, x:x + w]

    return trimmed_image


class CallCountDecorator:
    """
    A decorator that will count and print how many times the decorated function was called
    """

    def __init__(self, inline_func):
        """Create a decorator that counts calls to a function."""
        self.call_count = 0
        self.inline_func = inline_func

    def __call__(self, *args, **kwargs):
        """Count a call and then run the decorated function."""
        self.call_count += 1
        self._print_call_count()
        return self.inline_func(*args, **kwargs)

    def _print_call_count(self):
        """Print the current call count for the decorated function."""
        print(f" * The method {self.inline_func.__name__} called {self.call_count} times")


@CallCountDecorator
def transform_ecc(template_image: np.ndarray, input_image: np.ndarray) -> (float, np.ndarray, np.ndarray):
    """
    template_image: The "base" image
    input_image: The "target" image
    If possible, align input_image onto template_image
    Returns: cc, warp_matrix, input_image_aligned
    Source:  https://stackoverflow.com/questions/68497827/cv2-findtransformecc-how-to-ignore-small-particles
    """

    convMode = "down"  # "up"
    num_iterations = 1000
    corr_coeff = 1e-5  # 0.5

    # Find size of image1
    template_img_size = template_image.shape

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
    number_of_iterations = int(num_iterations)

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = float(corr_coeff)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(template_image, input_image, warp_matrix, warp_mode, criteria)
        # Warp the input_image to be similar to template_image
        # print(f'cc={cc}')
        assert cc >= MINIMUM_ACCEPTED_CC, "Correlation too low"
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            input_image_aligned = cv2.warpPerspective(input_image, warp_matrix,
                                                      (template_img_size[1], template_img_size[0]),
                                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            input_image_aligned = cv2.warpAffine(input_image, warp_matrix, (template_img_size[1], template_img_size[0]),
                                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

        cc = 1.0 - abs(1.0 - cc)  # Special fix: Handle case where cc>1 , wrap back from 1
        return cc, warp_matrix, input_image_aligned
    except cv2.error as e:
        return 0.0, None, template_image
    except AssertionError as e:
        return 0.0, None, template_image


@dataclass(order=True)  # , eq=False
class ImageCluster:
    """Store one group of similar subtitle character images.

    The purpose of this class is glyph normalization. It groups different
    images of the same character and stores enough data to make one example
    image for the group.

    ``ClusterManager`` first creates one cluster for each database image.
    Later, two clusters can be joined with ``+``. Their images are aligned
    with ECC and combined using their image counts.

    Attributes:
        total: Number of source images in the cluster.
        avg_cc: ECC similarity score from the latest merge. A new single-image
            cluster starts with ``0.0`` because it has not been compared yet.
        representative_id: Smallest database image ID in the cluster. It is
            used as the cluster's key.
        avg_img: Preprocessed grayscale image used for comparison and merging.
            Images must have compatible sizes.
        source_images: Database image IDs that belong to the cluster.

    This class only stores data in memory. It does not save clusters or
    average images in SQLite.
    """
    total: int
    avg_cc: float
    representative_id: int  # The "representative_id" of the group is the smallest id of all the images in the group
    avg_img: np.ndarray = field(compare=False)
    source_images: list = field(compare=False)

    default_transform_func = transform_ecc

    def __add__(self, other):
        """Merge two clusters and return their weighted average cluster."""
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


def weighted_average_of_images(img1, img2, weight1, weight2):
    """Return the weighted pixel average of two same-sized images."""
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


def get_file_attributes(file_name):
    """Return a file's name, parent directory, and creation date."""
    file_path = Path(file_name)
    return str(file_path.name), str(file_path.parent), get_file_date(file_name)


def create_combined_image_for_clusters(cluster1: ImageCluster, cluster2: ImageCluster, transform_func):
    """Align two cluster averages and combine them using cluster sizes."""
    if cluster1.avg_img.shape[0] < cluster2.avg_img.shape[0]:
        cluster1, cluster2 = cluster2, cluster1  # 1 is the larger image
    cc, warp_matrix, warped = transform_func(cluster1.avg_img,
                                             cluster2.avg_img)  # Project smaller onto larger image transform_ecc
    combined = weighted_average_of_images(cluster1.avg_img, warped, cluster1.total, cluster2.total)
    return cc, warp_matrix, combined


def get_file_date(file_name: Path):
    """Return a file's creation date, or ``None`` if the file is missing."""
    if os.path.exists(file_name):
        creation_timestamp = os.path.getctime(file_name)
        creation_datetime = datetime.datetime.fromtimestamp(creation_timestamp)
        return creation_datetime
    else:
        return None


def adjust_image_post_process(image):
    """Remove low values and blur an image for comparison or display."""
    _, image_blur = cv2.threshold(image, thresh=64, maxval=255, type=cv2.THRESH_TOZERO)
    # _, image_blur = cv2.threshold(image_blur, thresh=64, maxval=255, type=cv2.THRESH_TOZERO)
    # avg_blur = cv2.medianBlur(avg_blur, 15)
    image_blur = cv2.GaussianBlur(image_blur, (9, 9), 10)
    return image_blur


def pre_process_images(images, enlarge_ratio=None, border_size=BORDER_SIZE, invert=True):
    """Prepare images by converting, bordering, inverting, resizing, and blurring."""
    images = convert_images_to_greyscale_if_necessary(images)
    images_enlarged = [cv2.copyMakeBorder(  # Convert to Greyscale and add border
        img,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, None, value=255)
        for img in images]
    if invert:
        images_enlarged = invert_images(images_enlarged)
    if enlarge_ratio is not None:
        images_enlarged = [cv2.resize(img, None, fx=enlarge_ratio, fy=enlarge_ratio,
                                      interpolation=cv2.INTER_CUBIC)
                           for img in images_enlarged]
    images_enlarged = [adjust_image_post_process(img) for img in images_enlarged]
    return images_enlarged


def convert_images_to_greyscale_if_necessary(images):
    """Convert color images to grayscale and leave grayscale images unchanged."""
    if len(images[0].shape) == 3:  # Concert to Greyscale if not already
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    return images


def invert_images(images):
    """Invert each image so dark pixels become light and vice versa."""
    images = [255 - img for img in images]  # Convert to negative (White on Black)
    return images


def reverse_pre_process_images(images, enlarge_ratio=None):
    """Apply the source-image preparation steps used for display."""
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


def read_image_from_file_and_fix_aspect_ratio(sub_filename: str, aspect_ratio: float) -> Tuple[np.ndarray, float, float]:
    """Read an image, resize its width, and return the image and its size."""
    img_original = cv2.imread(sub_filename)
    hImg, wImg, _ = img_original.shape

    # Resize the image according to the aspect ratio correction.
    img_from_file_resized = cv2.resize(img_original, (int(aspect_ratio * wImg), int(hImg)))
    hImg, wImg, _ = img_from_file_resized.shape
    return img_from_file_resized, hImg, wImg
