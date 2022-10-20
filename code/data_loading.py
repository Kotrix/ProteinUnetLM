import os
from glob import glob
import numpy as np
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
from copy import deepcopy
output_folder = "../results"

# define problem properties
SS_LIST = ["C", "H", "E", "T", "G", "S", "I", "B"]
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
                      "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))
UPPER_LENGTH_LIMIT = 704


def read_fasta(filepath: str):
    # Read all non-empty lines from FASTA
    with open(filepath, 'r') as reader:
        lines = [line.strip() for line in reader if line.strip() != '']

    protein_names = []
    sequences = []
    new_sequence = True
    for line in lines:
        if line.startswith((">", ";")):
            protein_names.append(line[1:].strip())
            new_sequence = True
        elif new_sequence:
            sequences.append(line)
            new_sequence = False
        else:
            sequences[-1] = f"{sequences[-1]}{line}"
        
    return protein_names, sequences


def read_input_folder(folder_path: str):
    data_dict = defaultdict(dict)

    # Read FASTA files
    files = glob(os.path.join(folder_path, "*.fasta"))
    for file in files:
        protein_names, residue_lists = read_fasta(file)
        for protein_name, resnames in zip(protein_names, residue_lists):
            sequence = to_categorical([RESIDUE_DICT[residue] for residue in resnames], num_classes=NB_RESIDUES)
            data_dict[protein_name]["fasta"] = sequence
    
    # Read secondary structures
    files = glob(os.path.join(folder_path, "*.ss8"))
    for file in files:
        protein_names, residue_lists = read_fasta(file)
        for protein_name, ss8 in zip(protein_names, residue_lists):
            if protein_name not in data_dict:
                continue
            with open(os.path.join(output_folder, f"{protein_name}_true.ss8"), 'w') as file:
                file.write(">" + protein_name + "\n")
                file.write(ss8 + "\n")
            data_dict[protein_name]["true_ss8"] = np.array([t for t in ss8])

    print(len(data_dict), "proteins loaded")

    return data_dict


def standardize_data(data_dict: dict):

    mean = np.load(os.path.join("data_stats", "train_mean_prottrans.npy"))
    std = np.load(os.path.join("data_stats", "train_std_prottrans.npy"))

    for key in data_dict.keys():
        data_dict[key]["prottrans"] = (data_dict[key]["prottrans"] - mean) / std

    return data_dict


def fill_array_with_value(array: np.array, length_limit: int, value):

    filler = value * np.ones((length_limit - array.shape[0], array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array


def fill_with_zeros(data: dict, max_sequence_length: int):
    data_copy = deepcopy(data)
    for key, values in data_copy.items():
        if len(values) == UPPER_LENGTH_LIMIT or key == "true_ss8":
            continue
        data_copy[key] = fill_array_with_value(values, max_sequence_length, 0)

    return data_copy
