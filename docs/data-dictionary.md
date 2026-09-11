# Data dictionary for `benchmark_v1`

This dictionary describes the files under `data/benchmark_v1`. CSV and
Parquet files with the same stem contain the same logical columns. The SQLite
database is a relational copy of these package tables. Empty scale fields are
unavailable in the current export; they are not zero-valued measurements.

## Conventions and controlled vocabulary

| Term | Meaning |
|---|---|
| Source coordinate | Pixel coordinate in the representative source image; x increases right and y increases down. |
| Crop coordinate | Pixel coordinate in the 448 x 448 annotation-assisted axial-normalized image. |
| Normalized coordinate | Crop coordinate divided by 448; used by YOLO and the normalized crop table. |
| `source_csv_point_id` | Numeric point ID copied from the source CSV. It is provenance and is not a stable cross-format role ID. |
| `point_role` | `fish_landmark_source_order` or `scale_endpoint`. |
| `fish_source_order_id` | Fish landmark order 1--11 recovered from the fish YOLO row. |
| `scale_endpoint_id` | Stable package role ID 1--2 for a scale endpoint. |
| Canonical point | Package join key 1--11 with a provisional English name. |
| `farmed`, `wild` | Source origin labels; not independently reclassified here. |
| Candidate taxon | Source-derived field used for grouping, reporting and stratification; not an independent taxonomic determination. |
| `review_holdout` | Nine-image curation status for repeated-annotation disagreement; not part of the standard train/validation/test split. |

## Working landmark names

The package uses these names only as provisional cross-file labels pending
source-team point-by-point confirmation:

| Canonical ID | Provisional name |
|---:|---|
| 1 | `snout_tip` |
| 2 | `nape_dorsal_outline` |
| 3 | `dorsal_fin_origin` |
| 4 | `dorsal_fin_insertion` |
| 5 | `upper_caudal_fin_insertion` |
| 6 | `posterior_caudal_peduncle_midpoint` |
| 7 | `lower_caudal_fin_insertion` |
| 8 | `anal_fin_origin` |
| 9 | `pelvic_fin_origin` |
| 10 | `pectoral_fin_base` |
| 11 | `posterior_opercular_margin` |

These working labels do not constitute independent anatomical validation.

## Package-level files

| Path | Format | Rows/objects | Purpose |
|---|---|---:|---|
| `images/original/*.jpg` | JPEG | 2,564 files | Representative source-oriented images retaining source dimensions. |
| `yolo/images/{split}/*.jpg` | JPEG | 2,564 files | Generated 448 x 448 annotation-assisted axial-normalized views. |
| `yolo/labels/{split}/*.txt` | text | 2,564 files | One YOLO pose label per normalized image. |
| `yolo/data.yaml` | YAML | 1 config | One fish class, 11 keypoints, split paths and identity `flip_idx`. |
| `annotations/coco_keypoints.json` | JSON | 2,564 images; 2,564 annotations | COCO Keypoints export for normalized images. |
| `annotations/keypoints_crops_448.csv` / `.parquet` | tabular | 28,204 rows | Long-form canonical keypoints in crop pixels and normalized crop coordinates. |
| `annotations/crop_transforms.csv` / `.parquet` | tabular | 2,564 rows | Source-to-crop affine transform and normalization metadata. |
| `tables/benchmark_samples.csv` / `.parquet` | tabular | 2,564 rows | One row per curated benchmark image. |
| `tables/benchmark_keypoints.csv` / `.parquet` | tabular | 28,204 rows | One row per benchmark image and canonical point in source pixels. |
| `tables/keypoints_raw_long.csv` / `.parquet` | tabular | 34,944 rows | Source-derived raw points with role-aware fields. |
| `tables/keypoints_canonical_long.csv` / `.parquet` | tabular | source-derived | Source-record canonical mapping before exact-image consensus. |
| `tables/metadata.csv` / `.parquet` | tabular | 2,699 rows | Source-record inventory, parser status and QC metadata. |
| `tables/scale_endpoints_long.csv` / `.parquet` | tabular | 5,272 rows | Role-aware scale endpoints and blank physical-scale fields. |
| `tables/anomalies.csv` / `.parquet` | tabular | 272 rows | Parser and curation anomaly events. |
| `tables/exact_duplicates.csv` / `.parquet` | tabular | 231 rows | Source records in exact-content duplicate groups. |
| `tables/duplicate_annotation_agreement.csv` / `.parquet` | tabular | 104 rows | Repeated-annotation agreement summaries. |
| `tables/benchmark_exclusions.csv` / `.parquet` | tabular | 22 rows | Source records excluded for conflicting exact-image candidate taxa. |
| `tables/near_duplicate_review_candidates.csv` / `.parquet` | tabular | 27 rows | Perceptual near-duplicate candidates requiring review. |
| `splits/splits_standard.json` | JSON | 4 ID lists | Standard train/validation/test/review-holdout IDs. |
| `splits/splits_leave_one_taxon.json` | JSON | 9 fold objects | Candidate-taxon-held-out train/test ID lists. |
| `fish_landmarks.sqlite` | SQLite | 12 tables | Relational copy of the package tables. |
| `package_summary.json` | JSON | 1 object | Package counts and integrity-policy summary. |
| `file_manifest_no_hash.csv` | CSV | file list | Relative paths and byte sizes; no hash column. |

## Source metadata: `tables/metadata`

| Field | Type / unit | Definition |
|---|---|---|
| `sample_uid` | string | Join key for one source record. |
| `database_sample_id` | integer | Source database sample ID. |
| `source_group` | string | Source directory/group label. |
| `source_group_code` | string | Machine-readable source-group code from configuration. |
| `taxon_candidate` | string | Candidate scientific taxon derived from source-group mapping. |
| `origin_label_source` | string | Source origin label, such as `farmed` or `wild`. |
| `source_basename` | string | Source annotation/image basename. |
| `original_image` | path string | Source-relative original-image path. |
| `rendered_image` | path string | Source-relative rendered overlay path retained for provenance. |
| `csv_annotation` | path string | Source CSV annotation path. |
| `yolo_annotation` | path string | Source YOLO annotation path. |
| `width`, `height` | integer / px | Source image dimensions. |
| `file_bytes` | integer / bytes | Source image byte size used in curation metadata. |
| `content_id_blake2b64` | hexadecimal string | Cached 64-bit BLAKE2b content ID used for exact-image grouping. |
| `difference_hash_64` | hexadecimal string | Perceptual signature used for near-duplicate review. |
| `raw_point_count` | integer | Parsed non-zero raw point count. |
| `fish_landmarks_1_to_10_complete` | boolean | Whether fish role IDs 1--10 are complete. |
| `fish_landmarks_1_to_11_complete` | boolean | Whether fish role IDs 1--11 are complete. |
| `scale_endpoints_1_to_2_complete` | boolean | Whether both stable scale endpoint roles are present. |
| `legacy_raw_ids_1_to_11_present` | boolean | Whether legacy source CSV IDs 1--11 are present; not used as the role authority. |
| `legacy_raw_ids_12_to_13_present` | boolean | Whether legacy source CSV IDs 12--13 are present. |
| `yolo_row_count` | integer | Number of parsed YOLO rows. |
| `fish_yolo_nonzero_points` | number | Non-zero fish points reported by the fish YOLO row. |
| `scale_yolo_nonzero_points` | number | Non-zero scale points reported by a scale YOLO row, when present. |
| `csv_yolo_max_error_px` | number / px | Maximum source CSV-to-YOLO coordinate discrepancy for matched points. |
| `raw_polygon_signed_area` | number / px2 | Signed area of the parsed raw contour/order used for QC. |
| `body_axis_angle_deg` | number / degree | Legacy implementation field: angle of the role-1-to-role-6 working axial direction; not a validated morphometric body axis. |
| `canonical_mapping_mode` | controlled string | Current values include `direct`, `reverse`, `ambiguous` and `unresolved`. |
| `canonical_mapping_reason` | string | Reason recorded for the current role-aware mapping decision. |
| `canonical_mapping_complete` | boolean | Whether the record is sufficiently mapped for the current benchmark builder; it does not confirm anatomy. |
| `image_decode_error` | string | Image decoding error, blank when none was recorded. |
| `anomaly_count` | integer | Number of anomaly events for the source record. |
| `anomaly_codes` | pipe-delimited string | Compact anomaly-code list. |
| `exact_duplicate_group_id` | string | Exact-duplicate group label, blank when not applicable. |
| `is_exact_duplicate_member` | boolean | Whether the record belongs to an exact-duplicate group. |

## Curated image table: `tables/benchmark_samples`

There is one row per `benchmark_id`. Exact source duplicates within one
candidate taxon may contribute to one consensus row.

| Field | Type / unit | Definition |
|---|---|---|
| `benchmark_id` | string | Curated-image join key and image/label stem. |
| `content_id_blake2b64` | hexadecimal string | Exact-content grouping identity. |
| `representative_sample_uid` | string | Selected source record for the benchmark row. |
| `source_record_count` | integer | Number of source records collapsed into the row. |
| `source_sample_uids` | pipe-delimited string | Contributing source-record IDs. |
| `source_groups` | pipe-delimited string | Contributing source groups. |
| `taxon_candidate` | string | Candidate taxon used for grouping/stratification. |
| `origin_label` | string | Consolidated source origin; may be `conflicted`. |
| `origin_conflict` | boolean | Whether contributing records disagree on source origin. |
| `representative_original_image` | path string | Source image used for the packaged original copy. |
| `representative_rendered_image` | path string | Source rendered-overlay path retained as provenance. |
| `representative_csv_annotation` | path string | CSV used for the representative record. |
| `representative_yolo_annotation` | path string | YOLO annotation used for the representative record. |
| `width`, `height` | integer / px | Representative source dimensions. |
| `file_bytes` | integer / bytes | Representative source-image byte size. |
| `difference_hash_64` | hexadecimal string | Perceptual signature retained for review. |
| `scale_endpoints_1_to_2_complete` | boolean | Whether contributing records had both scale roles. |
| `canonical_mapping_modes` | pipe-delimited string | Mapping modes observed among contributors. |
| `canonical_mapping_reasons` | pipe-delimited string | Mapping reasons observed among contributors. |
| `provenance_status` | string | Current value is `source_team_confirmation_pending`. |
| `canonical_semantics_status` | string | Current value is `provisional_source_confirmation_pending`. |
| `cross_origin_eligible` | boolean | Whether this row is eligible for the cross-origin subset. |
| `repeated_annotation_mean_error_image_diagonal` | number / ratio | Mean repeated-annotation error divided by source-image diagonal. |
| `repeated_annotation_high_disagreement` | boolean | Whether the error exceeds 0.02 of the image diagonal. |
| `technical_validation_eligible` | boolean | False for the nine review-holdout rows. |
| `technical_validation_status` | controlled string | `eligible` or `review_holdout_repeated_annotation_disagreement`. |
| `split_stratum` | string | `taxon_candidate|origin_label` stratification key. |
| `standard_split` | controlled string | `train`, `validation`, `test` or `review_holdout`. |
| `leave_one_taxon_fold` | string | Candidate taxon used for the held-out evaluation fold. |

## Role-aware original-coordinate annotations

### `tables/keypoints_raw_long`

| Field | Type / unit | Definition |
|---|---|---|
| `sample_uid` | string | Source-record join key. |
| `source_group` | string | Source group label. |
| `source_csv_point_id` | integer | Original numeric point ID from the source CSV. |
| `point_role` | controlled string | `fish_landmark_source_order` or `scale_endpoint`. |
| `role_point_id` | integer | 1--11 for fish source order or 1--2 for scale endpoint role. |
| `x`, `y` | number / px | Source-image pixel coordinate. |

The fish role is assigned from the fish YOLO row and the scale role from the
scale YOLO row where available, with the source CSV ID retained separately.
In the four identified variant records, raw CSV IDs 11 and 12 are scale
endpoints and are not promoted to fish landmark roles.

### `tables/keypoints_canonical_long`

| Field | Type / unit | Definition |
|---|---|---|
| `sample_uid` | string | Source-record join key. |
| `canonical_id` | integer | Package join key 1--11. |
| `canonical_name` | string | Provisional name from the current configuration. |
| `x`, `y` | number / px | Source-image pixel coordinate. |
| `fish_source_order_id` | integer | Role-aware fish source order used for the mapping. |
| `source_csv_point_id` | integer | Original source CSV point ID retained for provenance. |

### `tables/benchmark_keypoints`

| Field | Type / unit | Definition |
|---|---|---|
| `benchmark_id` | string | Curated-image join key. |
| `canonical_id` | integer | 1--11. |
| `canonical_name_provisional` | string | Provisional cross-file label. |
| `x`, `y` | number / px | Median source-coordinate consensus. |
| `consensus_record_count` | integer | Number of source records contributing to this point. |
| `x_mad_px`, `y_mad_px` | number / px | Median absolute deviation of contributing coordinates. |

## Scale endpoints: `tables/scale_endpoints_long`

| Field | Type / unit | Definition |
|---|---|---|
| `sample_uid` | string | Source-record join key. |
| `scale_endpoint_id` | integer | Stable package role ID 1 or 2. |
| `source_csv_point_id` | integer | Original source CSV ID, commonly 12/13 in the dominant format and 11/12 in the four-point variant. |
| `x`, `y` | number / px | Source-image pixel coordinate. |
| `scale_length_value` | blank/number | Represented scale length when supplied; blank for all current rows. |
| `scale_unit` | blank/string | Unit for the represented scale length; blank for all current rows. |

## Crop annotations: `annotations/keypoints_crops_448`

| Field | Type / unit | Definition |
|---|---|---|
| `benchmark_id` | string | Curated-image join key. |
| `canonical_id` | integer | 1--11. |
| `canonical_name_provisional` | string | Provisional cross-file label. |
| `x_crop_px`, `y_crop_px` | number / px | Coordinate after the stored affine warp, clipped to the crop canvas. |
| `x_crop_normalized`, `y_crop_normalized` | number / ratio | Crop coordinate divided by 448. |
| `visibility` | integer | Exporter value `2` for every point; not an independently verified occlusion assessment. |

## Crop transforms: `annotations/crop_transforms`

| Field | Type / unit | Definition |
|---|---|---|
| `benchmark_id` | string | Curated-image join key. |
| `source_width`, `source_height` | integer / px | Source image dimensions. |
| `normalization_mode` | string | Current value `snout_left_dorsal_up_landmark_affine`. |
| `orientation_semantic_status` | string | Current value `provisional_pending_source_confirmation`. |
| `output_size` | integer / px | Current value 448. |
| `body_axis_length_px` | number / px | Distance between fish role points 1 and 6 in source pixels. |
| `axis_unit_x`, `axis_unit_y` | number | Unit vector along the provisional source-to-caudal axis. |
| `down_unit_x`, `down_unit_y` | number | Output-down unit vector selected by the geometric rule. |
| `u_min`, `u_max`, `v_min`, `v_max` | number | Landmark-derived axis extents before scaling/padding. |
| `affine_00`, `affine_01`, `affine_02`, `affine_10`, `affine_11`, `affine_12` | number | Entries of the 2 x 3 source-to-crop affine matrix. |
| `content_x0`, `content_y0`, `content_x1`, `content_y1` | number / crop px | Content-box bounds before clipping. |
| `resize_scale` | number / ratio | Scale applied by the affine warp. |

## Quality-control and exclusion tables

### `tables/anomalies`

| Field | Type | Definition |
|---|---|---|
| `sample_uid` | string | Source-record join key. |
| `source_group` | string | Source group. |
| `source_basename` | string | Source basename. |
| `anomaly_code` | string | Observed codes include `canonical_mapping_ambiguous`, `csv_point_count_0`, `csv_point_count_10`, `csv_point_count_11`, `csv_point_count_12`, `fish_landmark_count_0`, `fish_landmark_count_10`, `missing_fish_pose_row`, `missing_scale_row`, `nonpositive_raw_contour_orientation`, `scale_endpoint_count_0`, `source_variant_10_fish_plus_raw_11_12_scale` and `yolo_row_count_0`/`yolo_row_count_1`. |

### `tables/exact_duplicates`

| Field | Type | Definition |
|---|---|---|
| `duplicate_group_id` | string | Human-readable exact-duplicate group label. |
| `content_id_blake2b64` | hexadecimal string | Exact-content grouping ID. |
| `sample_uid` | string | Source-record member. |
| `group_size` | integer | Number of source records in the group. |

### `tables/duplicate_annotation_agreement`

| Field | Type / unit | Definition |
|---|---|---|
| `benchmark_id`, `content_id_blake2b64`, `taxon_candidate` | string | Curated-image, content and candidate-taxon identifiers. |
| `origin_conflict` | boolean | Whether source origins conflict. |
| `record_count`, `pair_count`, `complete_11_point_pair_count` | integer | Repeated-record and comparable-pair counts. |
| `pairwise_mean_error_px`, `pairwise_max_error_px` | number / px | Pairwise coordinate disagreement summaries. |
| `pairwise_mean_error_image_diagonal` | number / ratio | Mean pairwise error divided by image diagonal; 0.02 defines the review threshold. |

### `tables/benchmark_exclusions`

| Field | Type | Definition |
|---|---|---|
| `sample_uid` | string | Excluded source-record ID. |
| `content_id_blake2b64` | hexadecimal string | Conflicting exact-image content ID. |
| `source_group`, `taxon_candidate` | string | Source group and candidate taxon. |
| `exclusion_reason` | string | Current value `exact_image_conflicting_taxon_labels`. |

### `tables/near_duplicate_review_candidates`

| Field | Type / unit | Definition |
|---|---|---|
| `benchmark_id_left`, `benchmark_id_right` | string | Candidate benchmark-image pair. |
| `dhash_hamming_distance` | integer | Hamming distance between retained perceptual signatures. |
| `relative_file_size_difference` | number / ratio | Relative source-file-size difference. |
| `review_status` | string | Current value `manual_review_required`. |
| `automatic_action` | string | Current value `none`; no automatic filtering or grouping was applied. |

## Split JSON schemas

`splits_standard.json` contains `seed`, `ratios`, `stratification`, and arrays
named `train`, `validation`, `test` and `review_holdout`, each containing
`benchmark_id` strings. The current counts are 1,791, 382, 382 and 9.
`splits_leave_one_taxon.json` is keyed by candidate taxon; each value contains
`train` and `test` arrays of benchmark IDs.

## SQLite schema

`fish_landmarks.sqlite` contains these 12 tables:

```text
anomalies
benchmark_exclusions
benchmark_keypoints_crops_448
benchmark_keypoints_original
benchmark_samples
crop_transforms
duplicate_annotation_agreement
exact_duplicates
keypoints_raw_long
metadata
near_duplicate_review_candidates
scale_endpoints_long
```

The SQLite column names and types are the same logical fields documented
above. `benchmark_keypoints_original` is the SQLite copy of the
`tables/benchmark_keypoints` logical table, and
`benchmark_keypoints_crops_448` is the SQLite copy of the crop annotation
table.

## Units, semantics and release status

The 11 September 2026 operational interpretation is supplied separately in
`landmark-definitions-0911.csv`, joined by `canonical_id`. Frozen provisional
names are retained as historical aliases, not silently relabelled. In particular,
role 10 is a ventral head/opercular-boundary anchor and must not be treated as a
verified pectoral-fin attachment. Role 6 is a central caudal-base anchor, not an
arithmetic midpoint. No coordinates or frozen results changed in this update.

The team reports collection in the middle/lower Yangtze region. All 2,699 source
JPEGs and 2,564 released originals were checked and contained no EXIF tags;
capture date and equipment cannot be recovered from those files. This updates
the collection-level provenance only, not specimen-level location or batch.

All image-coordinate fields are pixels or normalized crop fractions. No
physical length is available because all current scale value/unit fields are
blank. Provisional point names, orientation convention, scale-side convention,
specimen-level collection/ethics fields and physical calibration are not
established by this release. Creator affiliations and repository metadata
are recorded in Zenodo record 22704999, version 1.0.1.
The team's generation and public-redistribution authority for the included
primary images and point annotations has been confirmed.
