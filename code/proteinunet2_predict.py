import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import torch
import tensorflow as tf
from transformers import T5Model, T5EncoderModel
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from tensorflow.keras.models import model_from_json
from data_loading import SS_LIST, UPPER_LENGTH_LIMIT, FASTA_RESIDUE_LIST, read_input_folder, standardize_data, fill_with_zeros
from architecture import unet_classifier
from custom_metrics import generate_summary

input_folder = "../data/inputs"
models_folder = "../data/models"
output_folder = "../results"


class OfflineProtTransT5XLU50Embedder(ProtTransT5XLU50Embedder):
    # Use an offline model directory
    def __init__(self, **kwargs):
        self.necessary_directories = []
        super().__init__(model_directory="../data/models/half_prottrans_t5_xl_u50", **kwargs)
        self._half_precision_model = True

    def get_model(self):
        if not self._decoder:
            model = T5EncoderModel.from_pretrained(self._model_directory, torch_dtype=torch.float16)
        else:
            model = T5Model.from_pretrained(self._model_directory, torch_dtype=torch.float16)
        return model


print("Loading ProtTrans model")   
embedder = OfflineProtTransT5XLU50Embedder()


def save_prediction_to_csv(one_hot: np.array, pred_c: np.array, true_c: np.array, protein_name: str):
    sequence_length = len(one_hot)

    output_df = pd.DataFrame()
    output_df["resname"] = [FASTA_RESIDUE_LIST[idx] for idx in np.argmax(one_hot, axis=-1)]

    def get_ss(one_hot):
        return [SS_LIST[idx] for idx in np.argmax(one_hot, axis=-1)]

    pred_ss8 = get_ss(pred_c[0])[:sequence_length]
    output_df["pred_SS8"] = pred_ss8

    if true_c is not None:
        with open(os.path.join(output_folder, f"{protein_name}_pred.ss8"), 'w') as file:
                file.write(">" + protein_name + "\n")
                file.write("".join(pred_ss8) + "\n")
        true_ss8 = [t for t in true_c]
        output_df["true_SS8"] = true_ss8
        generate_summary(protein_name, true_ss8, pred_ss8)

    output_df.to_csv(os.path.join(output_folder, protein_name + ".csv"), index=False)


def calculate_prottrans_features(data_dict: dict) -> dict:
    for protein_name in data_dict:
        sequence = "".join([FASTA_RESIDUE_LIST[idx] for idx in np.argmax(data_dict[protein_name]["fasta"], axis=-1)])
        data_dict[protein_name]["prottrans"] = embedder.embed(sequence)

    return data_dict


def main():
    print("Loading input data")
    data_dict = read_input_folder(input_folder)

    print("Calculating ProtTrans features")
    data_dict = calculate_prottrans_features(data_dict)

    print("Data standardization")
    data_dict = standardize_data(data_dict)

    print("Loading ProteinUnetLM model")
    tf.config.set_visible_devices([], 'GPU') # ProteinUnetLM is faster on CPU than on GPU
    model = unet_classifier()
    model.load_weights(os.path.join(models_folder, "ProteinUnetLM.h5"))
    model = tf.function(model) # speed up with graph execution

    for protein_name, data in data_dict.items():
        print("Folding protein", protein_name)

        protein_length = len(list(data.values())[0])
        if protein_length > UPPER_LENGTH_LIMIT:
            print(f"Sequence longer than {UPPER_LENGTH_LIMIT} residues are not supported!")
            continue

        filled_data = fill_with_zeros(data, UPPER_LENGTH_LIMIT)
        input_data = [filled_data["prottrans"], filled_data["fasta"]]
        input_data = [np.expand_dims(d, axis=0) for d in input_data]
        prediction = model(input_data)

        save_prediction_to_csv(data["fasta"], prediction, data.get("true_ss8", None), protein_name)


if __name__ == '__main__':
    main()
