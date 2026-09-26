# ExtractSubtitles: Architecture & System Overview

## Executive Summary

**ExtractSubtitles** is a Hebrew text extraction and recognition system that processes subtitle images from video files. It uses OCR (Tesseract), image alignment, and clustering algorithms to:
1. Extract Hebrew character images from subtitle JPEGs
2. Store them in a SQLite database
3. Group similar letter variations together through image registration
4. Generate canonical "average" glyphs for each Hebrew letter

The system is designed for optical character recognition (OCR) training, glyph normalization, and subtitle extraction workflows where consistent character representations are needed across typographic variations.

---

## High-Level Workflow

```
Video Subtitle Images (JPEG)
        ↓
[create_db.py] ← Tesseract OCR on JPEGs
        ↓
Extract individual character boxes + metadata
        ↓
Store in SQLite database (images + coordinates)
        ↓
[merge_letters.py] ← ClusterManager groups letter variants
        ↓
Image Registration (ECC alignment) + Correlation
        ↓
Hierarchical clustering merges similar glyphs
        ↓
Generate canonical letter images (averaged, aligned)
        ↓
Output: Normalized letter representations per Hebrew character
```

---

## Directory Structure

```
ExtractSubtitles/
├── src/                              # Core Python modules
│   ├── tesseract_hebrew.py           # Entry point (mostly deprecated) for OCR operations
│   ├── tesseract_hebrew_utils.py     # OCR utilities (bounding box extraction, image embedding)
│   ├── tesseract_sql.py              # Database layer (ORM-like dataclasses, SQLite adapter)
│   ├── create_db.py                  # Main pipeline: extract JPEGs → store in DB
│   ├── merge_letters.py              # Main pipeline: cluster letters → generate averages
│   ├── alignment.py                  # Image registration (ORB features, ECC transform)
│   ├── cluster.py                    # ClusterManager: hierarchical clustering engine
│   ├── utils.py                      # Shared utilities (image pre/post-processing, constants)
│   └── test_align.py                 # Test script for image alignment
│
├── notebooks/                        # Jupyter exploration notebooks
│   ├── FontAlign.ipynb               # Feature matching & homography test
│   ├── experiment_efficient_ecc.ipynb # ECC alignment experiments with distance matrices
│   └── experiment_efficient_ecc.py   # Parallel standalone version of notebook
│
├── resources/                        # Database schemas & data
│   ├── letters.sql                   # Database schema definition
│   ├── letters - Copy.sql            # Backup schema
│   ├── letters.sqlite                # SQLite database (main)
│   ├── letters - Copy.sqlite         # Backup DB
│   └── identifier.sqlite             # Empty (unused)
│
├── OCR_Samples/                      # Test subtitle images
│   └── *.jpeg                        # Sample frames from videos (8 samples)
│
├── distance_matrix_*.npy             # Cached correlation matrices (for letters ו, כ)
│
├── Pipfile                           # Python dependency spec (Python 3.12.2, currently empty)
│
└── ARCHITECTURE.md                   # This file
```

---

## Core Components

### 1. **Database Layer** (`tesseract_sql.py`)

**Purpose**: SQLite ORM abstraction with numpy array serialization.

**Key Classes**:
- `DatabaseManager`: Connection, initialization, CRUD operations
- `Image`: Represents a character glyph with:
  - `image_id` (PK)
  - `image_text` (Hebrew character, e.g., 'א')
  - `image` (numpy uint8 array, serialized to BLOB)
  - `image_hash` (MD5 deduplication)
  - `decoding_fk` (FK to subtitle metadata)
- `SubsDecoded`: Metadata linking images to subtitle files
- `SubsFiles`: Subtitle file metadata (path, date)
- `AspectCorrection`: Aspect ratio multipliers for image resizing

**Database Schema** (`resources/letters.sql`):
- Tables: `images`, `subs_decoded`, `subs_files`, `aspect_corrections`
- Numpy arrays stored as BLOB (serialized with `np.save`)

**Key Methods**:
```python
read_images_by_text_orderby_id(text)  # Query all instances of a letter
insert_image(image)                    # Insert or deduplicate by MD5 hash
insert_subs_decoded(...)               # Store OCR metadata
```

---

### 2. **Image Preprocessing & Utilities** (`utils.py`)

**Purpose**: Shared image transformations, constants, and helper functions.

**Thresholds** (tuning parameters for clustering):
```python
CORRELATION_THRESHOLD_FOR_MERGE = 0.94       # Min CC to merge clusters
CORRELATION_THRESHOLD_FOR_DISMISSAL = 0.7    # Min CC to keep cluster
FRACTION_OF_TOO_FAR_TO_ELIMINATE = 0.5       # Fraction of outliers to trigger elimination
BORDER_SIZE = 10                             # Padding around letters
```

**Key Functions**:
- `pre_process_images(images, enlarge_ratio, border_size, invert)`: Greyscale → border → invert (white-on-black) → scale → blur
- `transform_ecc(template, input)`: **Core alignment function** using OpenCV's `cv2.findTransformECC`
  - Computes homography matrix + correlation coefficient
  - Returns: `(cc, warp_matrix, aligned_image)`
  - Wrapped with `@CallCountDecorator` (tracks function calls)
- `ImageCluster`: Dataclass storing cluster state (total images, avg CC, representative ID, average image, source IDs)
- `trim_to_smallest_rectangle()`: Crop to content bounding box

---

### 3. **Image Alignment & Registration** (`alignment.py`)

**Purpose**: Feature extraction and homography computation for image alignment.

**Key Classes**:
- `FeatureExtraction`: ORB feature detector
  - Uses OpenCV's ORB keypoint detector (500 features, Harris scoring)
  - Stores keypoints, descriptors, drawn keypoints, matched points
- `AverageImageStat`: Statistics for a candidate average image (total count, avg CC, composite)

**Constants**:
```python
LOWES_RATIO = 0.7                  # Lowe's ratio test threshold
MIN_MATCHES = 2                    # Minimum matched features to proceed
```

**Key Functions**:
- `feature_matching(features0, features1)`: FLANN-based kNN descriptor matching
- `scale_convert_image(img, scale_percent)`: Resize grayscale image by percentage
- `calc_average_similar_base(base_index, images)`: ← **Deprecated** (see `ClusterManager` instead)
- `find_best_average_image_improved(images)`: Build correlation matrix, find best representative

---

### 4. **Clustering Engine** (`cluster.py`)

**Purpose**: Hierarchical agglomerative clustering for grouping letter variants.

**Core Class**: `ClusterManager`

**Workflow**:
1. Initialize: Each image = singleton cluster (correlation = 1.0)
2. **Main Loop** (`cluster_letters()`):
   - Pop two clusters from queue
   - Compute correlation coefficient (cached)
   - If CC > `CORRELATION_THRESHOLD_FOR_MERGE` (0.94) → merge
   - Check for outliers using `should_eliminate_cluster()` → mark as `None`
   - Re-queue for next round until convergence
3. **Merge Operation**:
   - Combine cluster metadata (average image, representative ID)
   - Update the correlation cache with the maximum known correlation from both clusters
   - Weighted average: new_avg = (img1 × count1 + img2 × count2) / (count1 + count2)
4. **Output**:
   - `self.clusters`: Dict of surviving clusters, sorted by size
   - `self.avg_image`: Trimmed, post-processed average of largest cluster

**Key Methods**:
```python
cluster_letters()              # Main clustering loop
merge_clusters(...)            # Combine two clusters
get_cc(distances_dict, ...)    # Retrieve (or compute & cache) correlation
should_eliminate_cluster(...)  # Outlier detection
merge_distances(...)           # Update distance metadata after merge
```

**Image Alignment Details**:
- Uses `transform_ecc()` (from `utils.py`) to register images
- Computes weighted average → post-processes (threshold, blur, trim)
- Displays visualization: source images + registered images + difference map

**Linkage policy**:
- The cache stores ECC correlation scores, where a larger value means greater similarity.
- After a merge, the implementation keeps the maximum correlation known against either source cluster.
- This is permissive and similar to single-linkage clustering: one similar member can keep two groups connected.
- A future alternative is to keep the minimum correlation. This would make clusters tighter and reduce chain merges, but it could split valid glyph variations caused by different fonts, sizes, or rendering noise.
- If the cache is changed to store distance values such as `1 - correlation`, the equivalent current policy is minimum distance.

---

### 5. **OCR Extraction Pipeline** (`create_db.py`)

**Purpose**: Main entry point for extracting letters from subtitle JPEG files.

**Entry Point**:
```python
python create_db.py <jpeg_location> <txt_location> <letters_location>
```

**Workflow**:
1. **Read JPEG files** from `jpeg_location`
2. **Apply aspect ratio correction** (from DB or constant 2.0)
3. **Run Tesseract OCR** with Hebrew config (`--oem 3 --psm 6 -l heb`)
4. **Extract bounding boxes** for each character using `pytesseract.image_to_boxes()`
5. **Enlarge boxes** by factor `BOX_ENLARGE_FACTOR = 1.2`
6. **Insert into database**:
   - Store character image (cropped from JPEG)
   - Store bounding box coordinates
   - Store subtitle file metadata
   - Cross-reference via FK

**Class**: `OcrBoxResult`
- Parses Tesseract box format: `<char> <left> <bottom> <right> <top> <page>`
- Converts Tesseract coords (Cartesian) to image coords (raster)
- Extracts box with enlargement

**Key Functions**:
- `perform_ocr_using_api_on_file_and_insert_into_db()`: Core ETL for one JPEG
- `image_to_boxes_keep_same()`: Wraps pytesseract to extract box coordinates

---

### 6. **Clustering Pipeline** (`merge_letters.py`)

**Purpose**: Main entry point for clustering and generating canonical letter images.

**Entry Point**:
```python
python merge_letters.py
```

**Hardcoded Parameters**:
```python
sqlite_db = CANONICAL_DATABASE_PATH  # Root-level letters.sqlite
LETTER = 'ל'  # Can be changed to other Hebrew letters (e.g., 'כ', 'ו')
```

**Workflow**:
1. Load all images for a letter from DB
2. Create `ClusterManager` instance
3. Run clustering algorithm
4. Display results:
   - Final cluster sizes
   - Largest clusters' average images (side-by-side)
   - Non-aligned outlier images (for manual review)

**Output**: 
- Console: Cluster statistics
- Visual: Image display windows
- DB: No changes (read-only, display-only)

---

### 7. **OCR Utilities** (`tesseract_hebrew_utils.py`)

**Purpose**: Image manipulation and Tesseract integration helpers.

**Tesseract Config**:
```python
TESSERACT_EXE = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Key Functions**:
- `embed_images_in_square(im_list, spacing)`: Create grid layout of images
- `hconcat_resize_max(im_list)`: Horizontal concatenation with height normalization
- `image_to_boxes_keep_same()`: Tesseract bounding box extraction
- `perform_ocr_commandline()`: Shell-based Tesseract invocation

**Deprecated Functions** (marked with `@deprecated`):
- `find_best_average_image()` → Use `ClusterManager` instead
- `new_char_filename()` → Filename generation logic

---

### 8. **Test & Experimental Scripts**

#### `test_align.py`
- **Purpose**: Verify alignment pipeline on sample data
- **Flow**: Load images → pre-process → embed → find best average → display
- Currently displays results but doesn't modify DB

#### `notebooks/FontAlign.ipynb`
- **Purpose**: Feature-based homography demonstration
- **Workflow**: ORB keypoint matching → homography computation → perspective warping
- Shows: matched features, warp matrix, aligned image
- **Status**: Experimental/exploratory

#### `notebooks/experiment_efficient_ecc.ipynb`
- **Purpose**: ECC-based alignment with distance matrix analysis
- **Experiments**: Correlation distributions, clustering heuristics
- **Artifacts**: Pre-computed `distance_matrix_*.npy` files for letters ו, כ
- **Status**: Data exploration notebook

---

## Execution Flow (Key Paths)

### Path 1: Extract Letters from Subtitles
```
create_db.py
├── Read JPEG files from OCR_Samples/
├── For each JPEG:
│   ├── tesseract_hebrew_utils.perform_ocr_using_api_on_file_and_insert_into_db()
│   ├── Tesseract OCR → bounding boxes
│   ├── OcrBoxResult: Parse boxes + enlarge
│   ├── tesseract_sql.DatabaseManager.insert_image()
│   └── tesseract_sql.DatabaseManager.insert_subs_decoded()
└── Database now contains character images + metadata
```

### Path 2: Cluster Letters → Generate Canonical Glyphs
```
merge_letters.py
├── tesseract_sql.DatabaseManager.read_images_by_text_orderby_id('ל')
├── cluster.ClusterManager(images_sql)
├── cluster_manager.cluster_letters()
│   ├── For each pair of clusters:
│   │   ├── utils.transform_ecc() → alignment + CC
│   │   ├── If CC > 0.94 → merge
│   │   ├── If cluster outlier → eliminate
│   │   └── Re-queue for next iteration
│   ├── Repeat until convergence
│   └── Post-process largest cluster → avg_image
└── Display results (visual verification, no DB writes)
```

---

## Abandoned / Experimental Code

### ❌ **Deprecated Components**

1. **`tesseract_hebrew.py`** (module-level)
   - Marked with `@deprecated` decorator
   - `main_old()` function is obsolete
   - Function `new_image_to_boxes()` → use Tesseract API directly instead

2. **`tesseract_hebrew_utils.py`**
   - `find_best_average_image()` → Use `ClusterManager.cluster_letters()` instead
   - `new_char_filename()` → Legacy filename generation

3. **Alignment Features in `alignment.py`**
   - `warp_image_1()` → Incomplete (feature-based homography; ECC is preferred)
   - `warp_image_2()` → Stub function (unused)
   - `calc_average_similar_base()` → Deprecated; ClusterManager handles this better

4. **Feature Matching** (`alignment.py`)
   - ORB feature detector is present but not used in main pipeline
   - ECC-based alignment (`transform_ecc`) is preferred for letter alignment

### ⚠️ **Experimental / Under-development**

1. **Notebook: `FontAlign.ipynb`**
   - Tests homography-based alignment (not the main pipeline)
   - Uses ORB features + FLANN matcher → ECC as fallback
   - Status: Exploratory; may inform future feature-based pipelines

2. **Notebook: `experiment_efficient_ecc.ipynb`**
   - Analyzes pre-computed distance matrices (`.npy` files)
   - Histogram distributions, best-match heuristics
   - Pre-computed matrices suggest this was an optimization study
   - Status: Analysis artifact; executable API names and project paths are updated, but it is not integrated into production flow

3. **`test_align.py`**
   - Minimal test harness; runs alignment on first 10 images
   - No assertions or formal test suite
   - Status: Manual debugging aid

4. **`test_align.py` references to `find_best_average_image_improved()`**
   - Calls deprecated function; should use `ClusterManager` instead

### 🔴 **Known Issues & Dead Code**

1. **Database selection**:
   ```python
   CANONICAL_DATABASE_PATH = PROJECT_ROOT / 'letters.sqlite'
   ```
   Active scripts now use the single canonical database path. The empty
   `letters2.sqlite` file is retained only as a historical artifact.

2. **Unused file**: `resources/identifier.sqlite` (empty)

3. **Unused/incomplete code in `create_db.py`**:
   - Lines 65-195: Extensive commented-out code for alternate extraction methods
   - Was probably earlier experimentation with different box extraction approaches

4. **Bug candidate** in `utils.py` line 186:
   ```python
   if cluster1.avg_img.shape[0] < cluster2.avg_img.shape[0]:
       cluster1, cluster2 = cluster1, cluster1  # ← Should be cluster2?
   ```
   → Might always assign cluster1 to itself (typo)

5. **Missing function reference** in `tesseract_hebrew_utils.py` line 164:
   ```python
   cv2.putText(output_img, text, (16, 16), fontFace=FONT_HERSHEY_PLAIN, ...)
   # ↑ FONT_HERSHEY_PLAIN is not imported; likely cv2.FONT_HERSHEY_PLAIN
   ```

6. **Unused import**: `Pipfile` lists Python 3.12.2 but includes no packages; dependencies are implicit

---

## Configuration & Tuning Parameters

### Image Processing (`utils.py`)
```python
ACCEPTABLE_EXTRA_DIFFERENCE_IN_DIMENSIONS = 0.3  # ±30% size variance allowed
CORRELATION_THRESHOLD_FOR_MERGE = 0.94           # Min CC to fuse clusters
CORRELATION_THRESHOLD_FOR_DISMISSAL = 0.7        # Min CC to keep cluster
FRACTION_OF_TOO_FAR_TO_ELIMINATE = 0.5           # Outlier elimination threshold
BORDER_SIZE = 10                                  # Padding (pixels)
MINIMUM_ACCEPTED_CC = 0.9                        # Minimum CC from ECC
```

### ECC Alignment (`utils.py` / `transform_ecc()`)
```python
num_iterations = 1000                # ECC iteration limit
corr_coeff = 1e-5                   # Convergence threshold
warp_mode = cv2.MOTION_HOMOGRAPHY   # Affine transformation
```

### ORB Features (`alignment.py`)
```python
nfeatures = 500                     # ORB detector: max features
scaleFactor = 1.5                   # ORB scale pyramid
scoreType = cv2.ORB_HARRIS_SCORE    # Harris corner weighting
```

### FLANN Matcher (`alignment.py`)
```python
LOWES_RATIO = 0.7                  # Lowe's ratio test threshold
MIN_MATCHES = 2                     # Minimum matches to proceed
```

### OCR (`create_db.py`)
```python
BOX_ENLARGE_FACTOR = 1.2            # Expand box by 20%
ASPECT_RATIO_CORRECTION = 2.0       # Resize multiplier for aspect correction
TESSERACT_CUSTOM_CONFIG_STR = r'--oem 3 --psm 6 -l heb'  # OCR config
```

---

## Mental Model for Returning Developer

### The Big Picture
This is a **Hebrew subtitle extraction and glyph normalization system**. Think of it as:

1. **Input**: Video subtitle frames (JPEGs) with Hebrew text rendered in various fonts
2. **Stage 1 (create_db.py)**: Extract each character's bounding box, store in SQLite
3. **Stage 2 (merge_letters.py)**: Group variants of the same letter, compute "canonical" average glyph
4. **Output**: Normalized, aligned letter images per Hebrew character

### Why It Matters
- **Tesseract OCR training**: Normalized glyphs improve training data quality
- **Font detection**: Identifies same letter across fonts/sizes by visual similarity
- **Subtitle processing**: Extracts character metadata for video indexing

### The Clustering Algorithm (Key Innovation)
- **Hierarchical agglomerative**: Starts with N clusters (one per image), merges if correlation > 0.94
- **Image registration**: Uses ECC (Enhanced Correlation Coefficient) to align before comparison
- **Iterative refinement**: Keeps processing queue until no new merges occur
- **Outlier elimination**: Marks clusters with too many low-correlation neighbors as invalid

### Practical Workflow
1. **To extract letters**: `python create_db.py OCR_Samples . .`
2. **To cluster & view**: `python merge_letters.py` (hardcode letter in LETTER variable)
3. **To inspect**: Open SQLite database with `sqlite3 letters.sqlite`

### Known Gotchas
- **Canonical database**: Active workflows use root-level `letters.sqlite`; `letters2.sqlite` is an empty historical artifact
- **Feature matching code**: Exists but isn't used; ECC is the workhorse
- **Experimental notebooks**: Lots of exploratory analysis; don't assume they match production code
- **Deprecated functions**: Several marked `@deprecated`; clean up if refactoring

### Code Quality Issues to Address
1. Magic numbers everywhere → Define more as module-level constants
2. Path handling → Use `pathlib.Path` consistently; avoid backslashes
3. Test coverage → Only exploratory notebooks; add unit tests
4. Error handling → Many bare exceptions; add meaningful logging
5. Type hints → Sparse; modernize with 3.12+ features

---

## Dependencies

### Runtime
- **pytesseract**: Tesseract OCR Python wrapper
- **opencv-python** (`cv2`): Image processing (alignment, features, transforms)
- **numpy**: Array operations, serialization
- **sqlite3**: Built-in; database access
- **sortedcontainers**: Efficient sorted dictionaries in clustering

### Development / Testing
- **jupyter**: Notebook environment (exploratory)
- **deprecation**: Decorator for deprecated functions

### External
- **Tesseract-OCR**: Command-line OCR engine (must be installed separately)
  - Path: `C:\Program Files\Tesseract-OCR\tesseract.exe` (hardcoded; adjust for your system)
  - Config: Hebrew language pack required (`-l heb`)

### Current Pipfile
- Lists Python 3.12.2 but has **empty** `[packages]` section
- Dependencies are implicit; should be formalized

---

## Recommendations for Resuming Development

### Immediate (Clarity)
1. [ ] Reconcile database files → Choose one canonical path
2. [ ] Fix hardcoded paths → Use `pathlib.Path` + config files
3. [ ] Document magic numbers → Move to `config.py` or module constants
4. [ ] Update `Pipfile` with explicit dependencies
5. [ ] Add logging → Replace print() with proper logger

### Short-term (Robustness)
1. [ ] Add unit tests → Test `transform_ecc`, clustering logic
2. [ ] Fix bug in `create_combined_image_for_clusters()` (line 186)
3. [ ] Fix missing import `FONT_HERSHEY_PLAIN`
4. [ ] Remove deprecated code or clearly version it
5. [ ] Add docstrings to main classes

### Medium-term (Features)
1. [ ] Integrate feature-based matching (ORB) as fallback for ECC failures
2. [ ] Implement proper test suite with pytest
3. [ ] Add CLI with argparse (replace hardcoded params)
4. [ ] Support batch processing of multiple letters
5. [ ] Export results (canonical images, metadata) to formats beyond SQLite

### Long-term (Architecture)
1. [ ] Consider moving to async/parallel processing for large datasets
2. [ ] Decouple DB from business logic → repository pattern
3. [ ] Add web UI for cluster review/adjustment
4. [ ] Performance profiling → ECC alignment is likely bottleneck
5. [ ] Support for other languages (currently Hebrew-specific)

---

## Summary Table

| Component | Status | Purpose | Entry Point |
|-----------|--------|---------|-------------|
| `create_db.py` | ✅ Active | Extract letters from JPEGs | `python create_db.py <paths>` |
| `merge_letters.py` | ✅ Active | Cluster & generate canonical images | `python merge_letters.py` |
| `cluster.py` | ✅ Active | Hierarchical clustering engine | `ClusterManager.cluster_letters()` |
| `utils.py` | ✅ Active | Shared utilities | Imported everywhere |
| `tesseract_sql.py` | ✅ Active | Database layer | `DatabaseManager()` |
| `alignment.py` | ⚠️ Partial | Image registration | `transform_ecc()` (via utils) |
| `test_align.py` | ⚠️ Test | Manual alignment test | `python test_align.py` |
| `tesseract_hebrew.py` | ❌ Deprecated | Legacy OCR entry point | None (use `create_db.py`) |
| `FontAlign.ipynb` | ⚠️ Experimental | Feature-based alignment exploration | Notebook; not production |
| `experiment_efficient_ecc.ipynb` | ⚠️ Experimental | Distance matrix analysis | Notebook; not production |

---

**Last Updated**: 2026-09-26 (Current HEAD)  
**Language**: Python 3.12.2  
**Primary Author**: AssafR
