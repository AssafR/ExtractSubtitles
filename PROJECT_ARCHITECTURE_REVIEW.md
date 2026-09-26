# ExtractSubtitles: Project Architecture Review

## Scope

This document reconstructs the project from its source code, SQL schema, notebooks, and the recommendations in `ARCHITECTURE.md`. It is an assessment of the current repository, not a claim about every original 2023 design intention.

No project files were changed as part of the analysis that produced this document.

## Executive Summary

ExtractSubtitles is a Hebrew subtitle glyph-extraction and normalization prototype. Its implemented workflow is:

```text
Subtitle JPEG frames
        |
        v
Tesseract Hebrew OCR
        |
        v
Character bounding boxes
        |
        v
Cropped glyph images + OCR metadata
        |
        v
SQLite database
        |
        v
OpenCV preprocessing and ECC alignment
        |
        v
Correlation-based glyph clustering
        |
        v
In-memory average glyphs displayed for inspection
```

The implemented boundary ends at visual inspection. There is no confirmed production path for subtitle-text export, Tesseract model training, persistent canonical glyphs, normalized subtitle generation, or a user-facing application.

The architecture described in `ARCHITECTURE.md` is broadly correct, but several paths, database names, APIs, and status claims are stale. The source code is the authority where it differs from the document.

## Main Execution Flow

### OCR ingestion

The active ingestion entry point is `src/create_db.py`.

1. The driver accepts an input image directory and two additional positional arguments.
2. It discovers files matching `*.jpeg`.
3. OpenCV loads each frame and applies aspect-ratio correction, currently defaulting to `2.0`.
4. `pytesseract` calls an external Tesseract installation with Hebrew configuration:

   ```text
   --oem 3 --psm 6 -l heb
   ```

5. `pytesseract.image_to_boxes()` returns character bounding boxes.
6. `OcrBoxResult` in `src/tesseract_hebrew_utils.py` parses Tesseract's bottom-origin coordinates into image coordinates.
7. Each character is cropped from the source image.
8. The glyph is inserted into SQLite as a NumPy array BLOB.
9. The character coordinates and source-file relationship are inserted into `subs_decoded`.

Important behavior:

- Only `.jpeg` files are matched; `.jpg` input is skipped.
- The enlarged bounding box is calculated, but the active persistence path appears to store the original crop.
- `txt_filename` and `letters_location` are passed through the active flow without producing active output.
- Database insertion results can be `None`, but callers dereference them immediately.
- The OCR script references `..\\resources\\letters4.sqlite`, while the checked-in database inventory is inconsistent with that path.

### Glyph clustering

The second active path begins in `src/merge_letters.py`.

1. A database is opened and glyphs for one hardcoded Hebrew letter are loaded.
2. `ClusterManager` creates one singleton cluster per image.
3. Images are converted, bordered, inverted, enlarged, thresholded, and blurred.
4. `transform_ecc()` uses OpenCV ECC registration to align image pairs.
5. Correlations above `0.94` cause clusters to merge.
6. Correlations below `0.7` contribute to outlier elimination.
7. Merged images become weighted averages.
8. The largest surviving clusters are displayed with OpenCV.

The clustering result is held in memory. It is not written back to SQLite.

## Technology Responsibilities

### Tesseract

Tesseract supplies Hebrew OCR text and per-character bounding boxes. `pytesseract` is only the Python wrapper. The Tesseract executable and Hebrew language data must be installed separately.

### OpenCV

OpenCV supplies image loading, resizing, aspect correction, grayscale conversion, borders, inversion, thresholding, blurring, ECC registration, ORB feature extraction, FLANN matching, and GUI display.

ECC is the alignment mechanism used by the current clustering route. ORB, FLANN, and homography code belong to an experimental route.

### SQLite

SQLite stores glyph arrays, OCR metadata, source-file metadata, and aspect-correction values. NumPy arrays are serialized with `numpy.save()` into BLOBs by `src/tesseract_sql.py`.

SQLite is not currently used to persist cluster membership, average glyphs, alignment matrices, or canonical glyph provenance.

## Database Model

The declared schema is in `resources/letters.sql`.

### `images`

Stores the detected character, serialized NumPy image, MD5 hash, and an optional `decoding_fk`.

`insert_image()` deduplicates globally by the MD5 hash of raw pixel bytes. Shape, dtype, source frame, and character identity are not included in the hash.

### `subs_files`

Stores source filename, directory, and filesystem timestamp. The combination is unique.

### `subs_decoded`

Stores the detected character, character index, page, bounding coordinates, source-file foreign key, and image foreign key. This is the relationship populated by the OCR ingestion flow.

### `aspect_corrections`

Stores known aspect-ratio correction values.

### `letters_mapping`

Appears intended for a later normalization or mapping stage. No Python code currently references it.

### Schema/application inconsistencies

The ingestion path inserts `subs_decoded.image_id_fk` but leaves `images.decoding_fk` null. The query method that accepts `from_subtitles=True` filters on `decoding_fk`, so its behavior is semantically ambiguous and likely incorrect.

The SQL schema declares foreign keys, but the Python connection does not clearly enable SQLite foreign-key enforcement with `PRAGMA foreign_keys = ON`.

Database and schema paths are process-relative and differ between scripts, notebooks, and tests. The repository therefore does not establish one reliable canonical database.

## Module Roles

- `src/create_db.py`: active OCR-to-SQL ingestion driver.
- `src/tesseract_sql.py`: SQLite connection, schema initialization, dataclasses, NumPy serialization, CRUD, and image lookup.
- `src/tesseract_hebrew_utils.py`: OCR box parsing, image layout helpers, and older averaging utilities.
- `src/utils.py`: image preprocessing, ECC registration, correlation handling, cluster-image arithmetic, and shared thresholds.
- `src/cluster.py`: heuristic agglomerative clustering and cluster visualization.
- `src/merge_letters.py`: hardcoded one-letter clustering/display driver.
- `src/alignment.py`: ORB/FLANN matching, homography experiments, and older ECC averaging logic.
- `src/test_align.py`: manual visual alignment harness, not an automated test suite.
- `src/tesseract_hebrew.py`: legacy OCR driver, apparently superseded by `create_db.py`.
- `resources/letters.sql`: declared database schema.
- `distance_matrix_*.npy`: cached research artifacts for ECC experiments.

## Notebook Research

### `notebooks/FontAlign.ipynb`

#### Research goal

This notebook investigates whether feature-based image registration can align glyphs before comparison or averaging.

#### Method

It experiments with:

- ORB keypoint detection
- FLANN descriptor matching
- Lowe's ratio test
- Homography estimation
- Perspective warping
- Visual comparison or blending of aligned images

#### Relationship to the project

The experiment addresses the same broad problem as ECC alignment, but it is not part of the current clustering path. Production clustering imports `transform_ecc()` from `utils.py`; it does not call the ORB/FLANN functions in `alignment.py`.

#### Current condition

The notebook uses an obsolete database method name, `read_images_by_text('ל')`, while the current implementation exposes `read_images_by_text_orderby_id()`. It also assumes a different relative database path. Its stored outputs are empty or incomplete, and no production code consumes a homography or generated image from it.

It should be treated as exploratory research rather than a reproducible current experiment.

### `notebooks/experiment_efficient_ecc.ipynb`

#### Research goal

This notebook investigates the computational cost and behavior of pairwise ECC comparison across many glyph images.

It appears to ask:

- What does the pairwise correlation distribution look like?
- Which glyph pairs are likely to be similar?
- Can expensive ECC comparisons be cached?
- How do forward and reverse alignment results differ?
- Which alignment failures or low-correlation pairs require inspection?

#### Method

The notebook uses NumPy and Pandas to inspect distance matrices, calculate descriptive statistics, generate histograms, select candidate pairs, and display aligned images and difference maps.

It loads `distance_matrix_כ.npy`, accesses glyphs from SQLite, and calls ECC alignment for visual investigation.

#### Relationship to the project

This research is related to the distance cache later maintained inside `ClusterManager`, but the production clustering code does not consume the checked-in `.npy` files. The matrices are therefore optimization and analysis artifacts, not production inputs.

#### Current condition

The notebook contains stale names such as `transform_ECC` and `read_images_by_text_orderbyid`, while the current source uses lowercase `transform_ecc` and a different query method name. It uses database paths that differ from the active scripts. Its cached matrices have no recorded database snapshot, preprocessing version, or threshold provenance.

The notebook contains historical outputs and execution counts, but those outputs should not be interpreted as a reproducible current run.

### `notebooks/experiment_efficient_ecc.py`

This standalone script is a more focused version of the distance-matrix experiment. It computes or reloads pairwise ECC correlations and uses current source names more consistently.

It is still not integrated into `ClusterManager`, has no clear executable entry point, and does not reproduce all of the notebook's analysis cells. It is best understood as research support code rather than a production component.

## Experimental, Obsolete, and Unused Code

### Experimental

- ORB and FLANN code in `src/alignment.py`
- Homography and perspective-warp experiments
- Both ECC distance-matrix notebooks
- `notebooks/experiment_efficient_ecc.py`
- Cached `distance_matrix_*.npy` files
- OpenCV display logic in `src/test_align.py`

### Deprecated or obsolete

- `src/tesseract_hebrew.py`
- `main_old()`
- `new_image_to_boxes()`
- `new_char_filename()`
- `find_best_average_image()`
- `calc_average_similar_base()`
- `warp_image_1()`
- `warp_image_2()`, which is effectively a stub

These represent earlier OCR, matching, or averaging approaches that were superseded by the current ingestion and `ClusterManager` paths.

### Unused or partially unused

- `letters_mapping` in the SQL schema
- `images.decoding_fk` in the active ingestion path
- `txt_filename`
- `letters_location`
- `get_cc_cache()` in `merge_letters.py`
- `create_tables()`
- `convert_array()`
- Cached distance matrices in production
- The optional image-label path using `FONT_HERSHEY_PLAIN`

## Confirmed Defects and Risks

1. Database and schema paths are inconsistent and depend on the working directory.
2. The active scripts refer to different database generations, so the intended canonical database is unclear.
3. Nullable database insertion results are dereferenced without checks.
4. The cluster image-size swap in `utils.py` assigns `cluster1` to itself instead of preserving `cluster2`.
5. Distance-cache references are not fully updated after clusters merge.
6. ECC failures are converted to `0.0`, which hides the reason for failure.
7. Homography-mode ECC may be unnecessarily flexible for small glyph images.
8. The clustering driver assumes a non-empty result and at least one multi-image cluster.
9. `test_align.py` references an averaging function from the wrong module.
10. `FONT_HERSHEY_PLAIN` is referenced without the required `cv2.` qualification.
11. The image hash does not include shape, dtype, source frame, or character identity.
12. There are no automated tests for coordinate conversion, database relationships, ECC failure behavior, cluster invariants, or average-glyph quality.
13. Notebook outputs and cached matrices lack dataset and preprocessing provenance.
14. Dependency configuration is split between `Pipfile`, `pyproject.toml`, and `uv.lock`.

## Reassessing the Recommendations in `ARCHITECTURE.md`

### Confirmed current issues

- Reconcile SQLite database files.
- Replace process-relative and Windows-specific paths.
- Add focused tests.
- Fix the cluster averaging typo.
- Fix the missing OpenCV font constant.
- Add validation around nullable database results.
- Add logging and meaningful error reporting.
- Replace hardcoded letters and database names with validated command-line arguments.
- Clarify or remove deprecated paths.

### Partially addressed

Dependencies are declared in `pyproject.toml`, but `Pipfile` is empty and environment metadata is inconsistent. Some configuration values are centralized in `utils.py`, but other thresholds and behavior remain distributed through the code. Some classes have docstrings, but the clustering and registration contracts are incomplete.

### Future enhancements

- Integrate ORB as an ECC fallback after measuring whether it improves registration.
- Batch-process multiple Hebrew letters.
- Persist canonical glyphs and their provenance.
- Export normalized glyphs outside SQLite.
- Profile ECC before introducing parallelism.
- Separate database access from image-processing logic.
- Build a review UI instead of relying on OpenCV windows.
- Generalize language and Tesseract configuration.

## Recommended Resumption Order

1. Establish one database and schema path, and verify the database contents before changing algorithms.
2. Repair the Python/SQL relationship contract, especially `decoding_fk`, `subs_decoded.image_id_fk`, and nullable insert results.
3. Add small tests for OCR coordinate conversion, image serialization, database relationships, and cluster merge invariants.
4. Fix the cluster-image swap and distance-cache update logic.
5. Make one reproducible command-line workflow for ingestion and one for clustering.
6. Re-run the ECC experiments against a documented database snapshot and preprocessing configuration.
7. Only then evaluate performance, ORB fallback, batch processing, or a persistent canonical-glyph export.

## Overall Assessment

The project contains a coherent research prototype with an active OCR ingestion path and an active in-memory glyph-clustering path. The central concepts are sound: use Tesseract for character localization, SQLite for glyph and provenance storage, and OpenCV ECC for visual registration.

The main obstacle to reliable resumption is not a missing algorithm. It is reproducibility and contract drift: divergent database paths, stale notebook APIs, incomplete cluster state updates, unchecked database results, and a lack of tests around image and database invariants.

The notebooks remain useful as historical research records. `FontAlign.ipynb` documents an alternative feature-based registration direction, while `experiment_efficient_ecc.ipynb` documents the investigation that motivated cached pairwise ECC comparisons. Neither is currently a reproducible production dependency.
