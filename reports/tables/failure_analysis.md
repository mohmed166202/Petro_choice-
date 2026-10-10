# Systematic Failure Case Analysis (15 Test Set Cases)

Diagnostic review of false negatives and false positives on the untouched test set.

| Case # | Image Name | Error Type | Ground Truth | Prediction | Root Cause | Proposed Fix | Impact Rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 000004.jpg | False Negative (Missed Detection) | hat | none | Dataset Annotation Noise (Missing Labels in Source) | Apply confident learning label cleaning (e.g. Cleanlab) to detect and flag unannotated heads in training data. | 6 |
| 2 | 000010.jpg | False Negative (Missed Detection) | hat | none | Dataset Annotation Noise (Missing Labels in Source) | Apply confident learning label cleaning (e.g. Cleanlab) to detect and flag unannotated heads in training data. | 6 |
| 3 | 000014.jpg | False Negative (Missed Detection) | hat | none | Partial Occlusion / Structural Obstruction | Incorporate temporal multi-frame tracking (ByteTrack / DeepSORT) to persist state across occluded frames. | 2 |
| 4 | 000021.jpg | False Positive (Background) | background | hat (0.90) | Headwear Ambiguity (Caps, Beanies, Hoodies) | Expand training annotations with negative hard mining of workers wearing non-PPE headgear. | 5 |
| 5 | 000027.jpg | False Negative (Missed Detection) | hat | none | Partial Occlusion / Structural Obstruction | Incorporate temporal multi-frame tracking (ByteTrack / DeepSORT) to persist state across occluded frames. | 2 |
| 6 | 000035.jpg | False Negative (Missed Detection) | hat | none | Partial Occlusion / Structural Obstruction | Incorporate temporal multi-frame tracking (ByteTrack / DeepSORT) to persist state across occluded frames. | 2 |
| 7 | 000055.jpg | False Positive (Background) | background | hat (0.92) | Headwear Ambiguity (Caps, Beanies, Hoodies) | Expand training annotations with negative hard mining of workers wearing non-PPE headgear. | 5 |
| 8 | 000061.jpg | False Negative (Missed Detection) | hat | none | Partial Occlusion / Structural Obstruction | Incorporate temporal multi-frame tracking (ByteTrack / DeepSORT) to persist state across occluded frames. | 2 |
| 9 | 000064.jpg | False Positive (Background) | background | hat (0.94) | Headwear Ambiguity (Caps, Beanies, Hoodies) | Expand training annotations with negative hard mining of workers wearing non-PPE headgear. | 5 |
| 10 | 000073.jpg | False Negative (Missed Detection) | hat | none | Partial Occlusion / Structural Obstruction | Incorporate temporal multi-frame tracking (ByteTrack / DeepSORT) to persist state across occluded frames. | 2 |
| 11 | 000078.jpg | False Negative (Missed Detection) | hat | none | Partial Occlusion / Structural Obstruction | Incorporate temporal multi-frame tracking (ByteTrack / DeepSORT) to persist state across occluded frames. | 2 |
| 12 | 000088.jpg | False Positive (Background) | background | hat (0.93) | Headwear Ambiguity (Caps, Beanies, Hoodies) | Expand training annotations with negative hard mining of workers wearing non-PPE headgear. | 5 |
| 13 | 000099.jpg | False Negative (Missed Detection) | hat | none | Partial Occlusion / Structural Obstruction | Incorporate temporal multi-frame tracking (ByteTrack / DeepSORT) to persist state across occluded frames. | 2 |
| 14 | 000106.jpg | False Negative (Missed Detection) | hat | none | Small / Distant Objects (<32px) | Deploy high-resolution tiling (SAHI / Slicing Aided Hyper Inference) or increase input size to 1024px. | 1 |
| 15 | 000109.jpg | False Negative (Missed Detection) | hat | none | Small / Distant Objects (<32px) | Deploy high-resolution tiling (SAHI / Slicing Aided Hyper Inference) or increase input size to 1024px. | 1 |

