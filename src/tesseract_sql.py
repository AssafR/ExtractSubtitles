import datetime
import pathlib
import sqlite3
import numpy as np
import io
import hashlib
import utils
from dataclasses import dataclass
from contextlib import closing
from typing import Optional

SQL_CREATION_SCRIPT = r'..\resources\letters.sql'

@dataclass
class AspectCorrection:
    """A class to represent an aspect ratio correction in the database."""
    aspect_correction_id: int
    aspect: float

    def __init__(self, aspect, aspect_correction_id=None):
        self.aspect = aspect
        self.aspect_correction_id = aspect_correction_id

    @classmethod
    def from_sql_query(cls, query_result):
        return cls(*query_result)


@dataclass
class Image:
    """A class to represent an image in the database."""

    image_id: int
    image_text: str
    image: np.ndarray  # Assuming the image is a numpy.ndarray
    image_hash: str  # New field
    decoding_fk: int = None

    # def __init__(self, text, img):
    #     self.image_id = None
    #     self.image_text = text
    #     self.image = img
    #     self.image_hash = self.calculate_image_hash()
    #
    def __init__(self, id, text, img, image_hash, decoding_fk=None):
        self.image_id = id
        self.image_text = text
        self.image = img
        self.image_hash = image_hash
        if not self.image_hash:
            self.image_hash = self.calculate_image_hash()
        self.decoding_fk = decoding_fk

    @classmethod
    def from_sql_query(cls, query_result):
        return cls(*query_result)

    def calculate_image_hash(self):
        # Calculate a hash of the image bytes (you can use different hashing algorithms)
        image_bytes = self.image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        return image_hash


@dataclass
class SubsDecoded:
    """A class to represent a decoded subtitle in the database."""
    subs_decoded_id: Optional[int]
    sub_file_fk: int
    image_id_fk: int
    detected_char: str
    char_index_in_text: int
    left: int
    right: int
    top: int
    bottom: int
    page: int

    @classmethod
    def from_sql_query(cls, query_result):
        return cls(*query_result)


@dataclass
class SubsFiles:
    """A class to represent a subtitle file in the database."""
    sub_file_id: Optional[int]
    file_name: str
    directory_name: str
    file_date: datetime.datetime

    def __init__(self, full_file_name):
        self.sub_file_id = None
        self.file_name, self.directory_name, self.file_date = utils.get_file_attributes(full_file_name)

    @classmethod
    def from_sql_query(cls, query_result):
        return cls(*query_result)

    def full_file_name(self):  # Manual addition
        return pathlib.Path(self.directory_name).joinpath(self.file_name).as_posix()


def create_connection(db_file_name):
    """ Create a database connection to a SQLite database."""
    db_file_abs_path = pathlib.Path(db_file_name).absolute()
    return sqlite3.connect(db_file_abs_path,
                           detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES  # For parsing datetypes
                           )


class DatabaseManager:
    """A class to manage the SQLite database."""

    def __init__(self, database):
        """ Create a connection to the database."""
        self.conn = create_connection(database)

        # Execute the SQL query creating the database tables if they don't exist
        with open(SQL_CREATION_SCRIPT, 'r') as file:
            sql_query = file.read()
            self.conn.executescript(sql_query)

        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        # sqlite3.register_converter("array", convert_array)
        sqlite3.register_converter("array", lambda x: np.load(io.BytesIO(x)))

    def create_tables(self, conn, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except sqlite3.Error as e:
            print(e)

    def read_aspect_corrections(self) -> list[AspectCorrection]:
        """ Read all aspect corrections from the database."""
        with closing(self.conn.cursor()) as cursor:
            cursor.execute("SELECT * FROM aspect_corrections")
            results = cursor.fetchall()
        return [AspectCorrection.from_sql_query(row) for row in results]

    def read_images(self) -> list[Image]:
        """ Read all images from the database."""
        with closing(self.conn.cursor()) as cursor:
            cursor.execute("SELECT * FROM images")
            results = cursor.fetchall()
        return [Image.from_sql_query(row) for row in results]

    def read_images_by_text_orderby_id(self, text, from_subtitles=True) -> list[Image]:
        # TODO: Document what subs_condition/from_subtitles does
        if from_subtitles:
            subs_condition = 'decoding_fk IS NULL'
        else:
            subs_condition = 'decoding_fk IS NOT NULL'

        with closing(self.conn.cursor()) as cursor:
            cursor.execute(f"SELECT * FROM images WHERE image_text=? AND {subs_condition} ORDER BY image_id ASC",
                           (text,))
            results = cursor.fetchall()
        return [Image.from_sql_query(row) for row in results]

    def read_subs_decoded(self) -> list[SubsDecoded]:
        """ Read all decoded subtitles from the database."""
        with closing(self.conn.cursor()) as cursor:
            cursor.execute("SELECT * FROM subs_decoded")
            results = cursor.fetchall()
        return [SubsDecoded.from_sql_query(row) for row in results]

    def read_subs_files(self) -> list[SubsFiles]:
        """ Read all subtitle files from the database."""
        with closing(self.conn.cursor()) as cursor:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM subs_files")
            results = cursor.fetchall()
        return [SubsFiles.from_sql_query(row) for row in results]

    def insert_aspect_correction(self, aspect_correction) -> Optional[AspectCorrection]:
        """ Insert an aspect correction into the database."""
        with closing(self.conn.cursor()) as cursor:
            cursor.execute("INSERT OR IGNORE INTO aspect_corrections (aspect) VALUES (?)", (aspect_correction.aspect,))
            self.conn.commit()
            cursor.execute('SELECT aspect_correction_id FROM aspect_corrections WHERE aspect = ?',
                           (aspect_correction.aspect,))
            result = cursor.fetchone()
        if result:
            aspect_correction.aspect_correction_id = result[0]
            return aspect_correction
        return None

    def insert_image(self, image) -> Optional[Image]:
        with closing(self.conn.cursor()) as cursor:
            # Construct a raw SQL statement for INSERT OR IGNORE and SELECT
            insert_sql = """
            INSERT OR IGNORE INTO images (image_text, image, image_hash, decoding_fk)
            VALUES (?, ?, ?, ?)
            """
            select_sql = """
            SELECT * FROM images WHERE image_hash = ?
            """

            # Execute the INSERT OR IGNORE statement
            cursor.execute(insert_sql, (image.image_text, image.image, image.image_hash, image.decoding_fk))
            self.conn.commit()

            # Query the database for the inserted or existing row
            result = cursor.execute(select_sql, (image.image_hash,)).fetchone()
        if result:
            # Close the cursor and return the row as an Image object
            return Image(*result)

        # If no existing row found, close the cursor and return None
        return None

    def insert_subs_decoded(self, subs_decoded: SubsDecoded) -> SubsDecoded:
        with closing(self.conn.cursor()) as cursor:
            cursor.execute(
                "INSERT INTO subs_decoded (sub_file_fk, image_id_fk, detected_char, char_index_in_text, [left], [right], top, bottom, page) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (subs_decoded.sub_file_fk, subs_decoded.image_id_fk, subs_decoded.detected_char,
                 subs_decoded.char_index_in_text, subs_decoded.left,
                 subs_decoded.right, subs_decoded.top, subs_decoded.bottom, subs_decoded.page))
            self.conn.commit()
            inserted_id = cursor.lastrowid
        subs_decoded.subs_decoded_id = inserted_id
        return subs_decoded

    def insert_subs_files(self, subs_files: SubsFiles) -> Optional[SubsFiles]:
        with closing(self.conn.cursor()) as cursor:
            cursor.execute("INSERT OR IGNORE INTO subs_files (file_name, directory_name, file_date) VALUES (?, ?, ?)",
                           (subs_files.file_name, subs_files.directory_name, subs_files.file_date))
            self.conn.commit()
            cursor.execute(
                "SELECT sub_file_id FROM subs_files WHERE file_name = ? AND directory_name = ? AND file_date = ?",
                (subs_files.file_name, subs_files.directory_name, subs_files.file_date))
            result = cursor.fetchone()
        if result:
            subs_files.sub_file_id = result[0]
            return subs_files
        return None


def adapt_array(arr):
    """
    SQLite does not have a storage type for arrays. This function converts a numpy array to a string of bytes.
    Converts np.array to TEXT when inserting
    Source: https://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.getvalue())


def convert_array(text):
    """
    SQLite does not have a storage type for arrays. This function converts a string of bytes to a numpy array.
    Converts TEXT to np.array when selecting
    Source: https://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def read_images_for_letter(db, letter, from_subtitles=True) -> list[np.ndarray]:
    """ Read images for a given letter from the database and apply pre_process to them """
    images_sql = db.read_images_by_text_orderby_id(letter, from_subtitles)
    # images_sql = sorted(images_sql, key=lambda x: x.image_id, reverse=False)
    images_raw = [img.image for img in images_sql]
    images = utils.pre_process_images(images_raw, 3.0)
    return images

#

# UPDATE images
# SET decoding_fk = subs_decoded.subs_decoded_id
# FROM subs_decoded
# WHERE images.image_id = subs_decoded.image_id_fk;
