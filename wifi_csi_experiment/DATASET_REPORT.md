# DATASET REPORT: WiAR Wi-Fi CSI Dataset

## 1. Overview
This report documents the structural audit and verification of the **WiAR Wi-Fi CSI Dataset** used for evaluating deep learning models in Wi-Fi CSI-based Human Activity Recognition (HAR).

- **Raw Dataset Root Directory**: `WiAR-master/data`
- **Total Archives / Top-Level Subdirectories**: 12 folders
- **Total Identified Subjects / Volunteers**: 10 distinct individuals
- **Total Dataset Files / Samples**: 4161 samples
- **Total Activity Classes**: 16 activities (`a1` to `a16`)

---

## 2. Identified Volunteers & Folder Mapping

The raw archives contain 12 folders which map to 10 distinct subject identities:

| Subject ID | Source Folders | Total Samples | Activities Present | Samples per Activity |
|---|---|---|---|---|
| `volunteer_7` | `volunteer_7` | 480 | 16 (`a1`-`a16`) | 30 |
| `volunteer_8` | `volunteer_8` | 480 | 16 (`a1`-`a16`) | 30 |
| `volunteer_9` | `volunteer_9` | 480 | 16 (`a1`-`a16`) | 30 |
| `volunteer_10` | `volunteer_10` | 480 | 16 (`a1`-`a16`) | 30 |
| `volunteer_a` | `volunteer_a_all_data` | 481 | 16 (`a1`-`a16`) | 30 (31 for `a5`) |
| `volunteer_b` | `volunteer_b_all_data` | 480 | 16 (`a1`-`a16`) | 30 |
| `volunteer_c` | `volunteer_c_all_data` | 480 | 16 (`a1`-`a16`) | 30 |
| `volunteer_d` | `volunteerD_1`, `volunteerD_2` | 320 | 16 (`a1`-`a16`) | 20 (`a1`-`a11` in part 1, `a12`-`a16` in part 2) |
| `volunteer_e` | `volunteerE_1`, `volunteerE_2` | 320 | 16 (`a1`-`a16`) | 20 (`a1`-`a10` in part 1, `a11`-`a16` in part 2) |
| `volunteer_f` | `volunteer_f_all_data` | 160 | 16 (`a1`-`a16`) | 10 |

---

## 3. Activity Mapping

The dataset contains 16 human activity classes:
1. `a1`: Horizontal arm wave
2. `a2`: Vertical arm wave
3. `a3`: Draw circle (clockwise)
4. `a4`: Draw circle (counterclockwise)
5. `a5`: Draw tick
6. `a6`: Draw cross
7. `a7`: Come here gesture
8. `a8`: Wave gesture
9. `a9`: Bend
10. `a10`: Squat
11. `a11`: Walk
12. `a12`: Fall
13. `a13`: Run
14. `a14`: Sit down
15. `a15`: Stand up
16. `a16`: Pick up

---

## 4. File Naming & CSI Payload Format

- **Filename Structure**: `csi_a{activity_id}_{sample_id}.dat` (or without `.dat` extension in Volunteer D and E folders).
- **CSI Parsing Tool**: `csiread.Intel` (Intel 5300 802.11n NIC format).
- **Raw CSI Data Shape per Sample**: `(Npackets, 30 subcarriers, 3 Rx, 2 Tx)`
- **Total Channels per Packet**: 30 * 3 * 2 = 180 subcarrier channels.
- **Packet Length Variation**: Samples typically contain between 200 and 600 packets depending on activity duration.

---

## 5. Technical Observations & Ambiguities Identified

1. **Extension Variations**: Files under `volunteerD_1`, `volunteerD_2`, `volunteerE_1`, `volunteerE_2` do not carry explicit `.dat` file extensions, though their internal binary format is identical Intel 5300 CSI payload.
2. **Folder Splitting**: Volunteers D and E were collected across two separate subfolders (`volunteerD_1` + `volunteerD_2` and `volunteerE_1` + `volunteerE_2`). These must be merged into single volunteer subject entities `volunteer_d` and `volunteer_e` to prevent data duplication or misclassification.
3. **Extra Sample**: `volunteer_a` contains 31 trials for activity `a5` (`csi_a5_31.dat`), whereas all other activities have 30.
4. **Subject Sample Heterogeneity**: Volunteers 7, 8, 9, 10, A, B, C have 30 samples/activity; Volunteers D, E have 20 samples/activity; Volunteer F has 10 samples/activity.

---
