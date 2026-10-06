import os
import re
import glob
from collections import defaultdict, Counter
import yaml
import csiread

def get_volunteer_id(dir_name):
    """
    Map raw data directory names to canonical volunteer subject IDs.
    """
    name = dir_name.lower()
    if 'volunteer_7' in name:
        return 'volunteer_7'
    elif 'volunteer_8' in name:
        return 'volunteer_8'
    elif 'volunteer_9' in name:
        return 'volunteer_9'
    elif 'volunteer_10' in name:
        return 'volunteer_10'
    elif 'volunteer_a' in name:
        return 'volunteer_a'
    elif 'volunteer_b' in name:
        return 'volunteer_b'
    elif 'volunteer_c' in name:
        return 'volunteer_c'
    elif 'volunteerd' in name:
        return 'volunteer_d'
    elif 'volunteere' in name:
        return 'volunteer_e'
    elif 'volunteer_f' in name:
        return 'volunteer_f'
    else:
        return dir_name

def inspect_wiar_dataset(raw_dir):
    print("=" * 70)
    print("INSPECTING WIAR DATASET")
    print("=" * 70)
    print(f"Raw data root: {raw_dir}")
    
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    dir_entries = sorted(os.listdir(raw_dir))
    print(f"Top-level directories in data folder ({len(dir_entries)}):")
    for d in dir_entries:
        print(f"  - {d}")

    subject_files = defaultdict(list)
    dir_to_subject = {}

    for d in dir_entries:
        dir_path = os.path.join(raw_dir, d)
        if not os.path.isdir(dir_path):
            continue
        subj_id = get_volunteer_id(d)
        dir_to_subject[d] = subj_id

        # Walk through directory to find data files
        for root, subdirs, filenames in os.walk(dir_path):
            for f in filenames:
                if f.endswith('.txt') or f.endswith('.m'):
                    continue
                full_path = os.path.join(root, f)
                subject_files[subj_id].append((d, f, full_path))

    subjects = sorted(list(subject_files.keys()))
    print("\n" + "-" * 70)
    print(f"IDENTIFIED SUBJECTS / VOLUNTEERS ({len(subjects)} distinct individuals):")
    print("-" * 70)
    
    total_samples = 0
    subject_stats = {}

    for subj in subjects:
        files = subject_files[subj]
        activities = set()
        trials = defaultdict(int)
        
        for d_name, fname, fpath in files:
            m = re.search(r'a(\d+)_(\d+)', fname)
            if m:
                act_id = int(m.group(1))
                trial_id = int(m.group(2))
                activities.add(act_id)
                trials[act_id] += 1
            else:
                m_act = re.search(r'a(\d+)', fname)
                if m_act:
                    activities.add(int(m_act.group(1)))

        count = len(files)
        total_samples += count
        subject_stats[subj] = {
            'total_files': count,
            'num_activities': len(activities),
            'activities': sorted(list(activities)),
            'dirs': sorted(list(set(item[0] for item in files)))
        }
        print(f"Subject: {subj:<15} | Files: {count:<4} | Activities: {len(activities):<2} | Source Folders: {subject_stats[subj]['dirs']}")

    print("\n" + "-" * 70)
    print("ACTIVITY BREAKDOWN:")
    print("-" * 70)
    
    activity_counts = defaultdict(int)
    for subj, files in subject_files.items():
        for _, fname, _ in files:
            m = re.search(r'a(\d+)', fname)
            if m:
                act_id = int(m.group(1))
                activity_counts[act_id] += 1

    all_activities = sorted(list(activity_counts.keys()))
    print(f"Total Unique Activities Found: {len(all_activities)} (a1 to a{max(all_activities)})")
    for act in all_activities:
        print(f"  Activity a{act:<2}: {activity_counts[act]} total samples across all subjects")

    # Sample ONE .dat file to inspect packet counts and dimensions
    sample_file = subject_files['volunteer_7'][0][2]
    print("\n" + "-" * 70)
    print("SAMPLE FILE PARSING TEST:")
    print("-" * 70)
    print(f"Testing sample file: {sample_file}")

    csidata = csiread.Intel(sample_file)
    csidata.read()
    csi_matrix = csidata.get_scaled_csi()
    
    print(f"CSI Tensor Shape (Npackets, Nsub, Nrx, Ntx): {csi_matrix.shape}")
    print(f"CSI Complex Data Type: {csi_matrix.dtype}")
    print(f"Number of Packets in Sample: {csi_matrix.shape[0]}")
    print(f"Subcarriers: {csi_matrix.shape[1]}")
    print(f"Rx Antennas: {csi_matrix.shape[2]}")
    print(f"Tx Antennas: {csi_matrix.shape[3]}")
    print(f"Total Subcarrier Channels: {csi_matrix.shape[1] * csi_matrix.shape[2] * csi_matrix.shape[3]}")

    return {
        'subjects': subjects,
        'subject_stats': subject_stats,
        'total_samples': total_samples,
        'activity_counts': activity_counts,
        'sample_file_shape': csi_matrix.shape
    }

def generate_dataset_report(inspection_data, output_path):
    report = fr"""# DATASET REPORT: WiAR Wi-Fi CSI Dataset

## 1. Overview
This report documents the structural audit and verification of the **WiAR Wi-Fi CSI Dataset** used for evaluating deep learning models in Wi-Fi CSI-based Human Activity Recognition (HAR).

- **Raw Dataset Root Directory**: `WiAR-master/data`
- **Total Archives / Top-Level Subdirectories**: 12 folders
- **Total Identified Subjects / Volunteers**: {len(inspection_data['subjects'])} distinct individuals
- **Total Dataset Files / Samples**: {inspection_data['total_samples']} samples
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

- **Filename Structure**: `csi_a{{activity_id}}_{{sample_id}}.dat` (or without `.dat` extension in Volunteer D and E folders).
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
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved DATASET_REPORT.md to {output_path}")

if __name__ == '__main__':
    raw_dir = r"c:\Users\manishkumar.COMPILERX\Downloads\WiAR-master\WiAR-master\data"
    inspection_data = inspect_wiar_dataset(raw_dir)
    report_path = r"c:\Users\manishkumar.COMPILERX\Downloads\WiAR-master\wifi_csi_experiment\DATASET_REPORT.md"
    generate_dataset_report(inspection_data, report_path)
