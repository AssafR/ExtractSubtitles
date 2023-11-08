--
-- File generated with SQLiteStudio v3.4.4 on ??? ? ??? 5 20:47:41 2023
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
    image_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    image_text TEXT,
    image      ARRAY
);


-- Table: subs_decoded
DROP TABLE IF EXISTS subs_decoded;

CREATE TABLE IF NOT EXISTS subs_decoded (
    subs_decoded_id INTEGER PRIMARY KEY,
    sub_file_fk             REFERENCES subs_files (sub_file_id) 
                            NOT NULL,
    image_id_fk     INTEGER REFERENCES images (image_id) 
                            NOT NULL,
    detected_char   TEXT,
    [left]          INTEGER NOT NULL,
    [right]         INTEGER NOT NULL,
    top             INTEGER NOT NULL,
    bottom          INTEGER NOT NULL,
    page            INTEGER NOT NULL
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
