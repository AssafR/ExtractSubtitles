--
-- File generated with SQLiteStudio v3.4.4 on ??? ? ??? 30 22:54:19 2023
--
-- Text encoding used: System
--
PRAGMA foreign_keys = off;
BEGIN TRANSACTION;

-- Table: aspect_corrections
DROP TABLE IF EXISTS aspect_corrections;

CREATE TABLE IF NOT EXISTS aspect_corrections (
    aspect_correction_id INTEGER PRIMARY KEY ASC ON CONFLICT ABORT AUTOINCREMENT,
    aspect               REAL    NOT NULL
                                 UNIQUE ON CONFLICT IGNORE
);


-- Table: images
DROP TABLE IF EXISTS images;

CREATE TABLE IF NOT EXISTS images (
    image_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    image_text  TEXT,
    image       ARRAY,
    image_hash  TEXT    UNIQUE ON CONFLICT IGNORE,
    decoding_fk INTEGER REFERENCES subs_decoded (subs_decoded_id) 
);


-- Table: letters_mapping
DROP TABLE IF EXISTS letters_mapping;

CREATE TABLE IF NOT EXISTS letters_mapping (
    mapping_id         INTEGER PRIMARY KEY ASC ON CONFLICT ROLLBACK AUTOINCREMENT
                               UNIQUE
                               NOT NULL,
    detected_char      TEXT,
    subs_decoded_id_fk         REFERENCES subs_decoded (subs_decoded_id) 
                               NOT NULL,
    image_id_fk                REFERENCES images (image_id) 
);


-- Table: subs_decoded
DROP TABLE IF EXISTS subs_decoded;

CREATE TABLE IF NOT EXISTS subs_decoded (
    subs_decoded_id    INTEGER PRIMARY KEY,
    sub_file_fk                REFERENCES subs_files (sub_file_id) 
                               NOT NULL,
    image_id_fk        INTEGER REFERENCES images (image_id) 
                               NOT NULL,
    detected_char      TEXT,
    char_index_in_text INTEGER,
    [left]             INTEGER NOT NULL,
    [right]            INTEGER NOT NULL,
    top                INTEGER NOT NULL,
    bottom             INTEGER NOT NULL,
    page               INTEGER NOT NULL
);


-- Table: subs_files
DROP TABLE IF EXISTS subs_files;

CREATE TABLE IF NOT EXISTS subs_files (
    sub_file_id    INTEGER   PRIMARY KEY,
    file_name      TEXT      CONSTRAINT sub_files_NotNull NOT NULL,
    directory_name TEXT,
    file_date      TIMESTAMP NOT NULL,
    UNIQUE (
        file_name,
        directory_name,
        file_date
    )
    ON CONFLICT IGNORE
);


COMMIT TRANSACTION;
PRAGMA foreign_keys = on;
