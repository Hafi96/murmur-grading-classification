import numpy as np
import os
import torch
from helper_code import find_patient_files, load_patient_data, get_num_locations, get_locations, get_grade, load_wav_file
from helper_code import compare_strings, get_age, get_sex, get_height, get_weight, get_pregnancy_status


grade_mapping_str2int = {'I/VI': 0, 'II/VI': 1,'III/VI': 2}
grade_mapping_int2str = {0: 'I/VI', 1: 'II/VI',2: 'III/VI'}

age_wise_avg_height_dict = {
    'Neonate': 49.33,
    'Infant': 63.29,
    'Child': 114.91,
    'Adolescent': 153.73,
    'Young Adult': 175.00,
    'nan': 110.80
}

age_wise_avg_weight_dict = {
    'Neonate': 3.42,
    'Infant': 7.40,
    'Child': 23.94,
    'Adolescent': 50.08,
    'Young Adult': 60.00,
    'nan': 23.63
}

def convert_label_to_int(str_label):
    return grade_mapping_str2int[str_label]

def convert_label_to_str(int_label):
    return grade_mapping_int2str[int_label]

def get_grade_locations(data):
    locations = None
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            locations = l.split(': ')[1].split('+')
            break
    if locations is None:
        raise ValueError('No murmur location available!')
    return locations

def get_patient_recording_files(data, num_locations):
    recording_files = []
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i == 0:
            pass
        elif 1 <= i <= num_locations:
            recording_files.append(entries[2])
        else:
            break
    return recording_files

def load_recordings_with_labels(data_folder, included_labels=["I/VI","II/VI","III/VI"]):   
    patient_files_arr, recording_files, grades = [], [], []
    patient_files = find_patient_files(data_folder)
    
    for pf in patient_files:
        patient_data = load_patient_data(pf)
        patient_grade = get_grade(patient_data)

        # Skip patients whose murmur grade is not in the included labels
        if patient_grade not in included_labels:
            continue
        
        patient_grade_idx = included_labels.index(patient_grade)  # Assign index-based labels
        locations = get_locations(patient_data)
        grade_locations = get_grade_locations(patient_data)
        p_recordings = get_patient_recording_files(patient_data, len(locations))

        # Assign labels based on location rules
        for i in range(len(locations)):
            if included_labels[patient_grade_idx] == "I/VI" and locations[i] not in grade_locations:
                continue
            elif included_labels[patient_grade_idx] == "II/VI" and locations[i] not in grade_locations:
                continue
            elif included_labels[patient_grade_idx] == "III/VI" and locations[i] not in grade_locations:
                continue
            else:
                patient_files_arr.append(pf)
                recording_files.append(os.path.join(data_folder, p_recordings[i]))
                grades.append(patient_grade_idx)

    # Convert lists to numpy arrays for better handling
    patient_files_arr = np.array(patient_files_arr, dtype=np.str_)
    recording_files = np.array(recording_files, dtype=np.str_)
    grades = np.array(grades, dtype=np.int_)

    return patient_files_arr, recording_files, grades

def load_patient_features(data):    
    age_group = get_age(data)
    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6.
    elif compare_strings(age_group, 'Child'):
        age = 6. * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15. * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20. * 12
    else:
        age = 6. * 12  # Default to child age if unspecified

    sex_features = np.zeros(2, dtype=np.float32)
    sex = get_sex(data)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1
    
    pregnancy = 1. if get_pregnancy_status(data) else 0.
    
    height = np.float32(get_height(data))
    weight = np.float32(get_weight(data))
    if np.isnan(height): 
        height = np.float32(age_wise_avg_height_dict[age_group])
    if np.isnan(weight): 
        weight = np.float32(age_wise_avg_weight_dict[age_group])
    
    # Number of features: 1 + 2 + 1 + 1 + 1 = 6
    features = np.hstack(([age], sex_features, [height], [weight], [pregnancy])).astype(np.float32)
    return features

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_accuracy(output, target):
    batch_size = target.size(0)
    _, pred = torch.max(output, dim=1)
    n_correct = (pred == target).sum()
    acc = n_correct / batch_size
    return acc
