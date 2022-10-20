import tensorflow as tf
tf.random.set_seed(1992)
from tensorflow.keras import backend as K
import os
import subprocess
import numpy as np
from pycm import ConfusionMatrix
from sklearn.metrics import f1_score, matthews_corrcoef

PATH_TO_SOV_SCRIPT = "SOV_refine.pl"
output_folder = "../results"
MASK_VALUE = 9999

def masked_acc(y_true, y_pred):
    mask = tf.reduce_any(tf.not_equal(y_true, MASK_VALUE), -1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return K.mean(K.equal(K.argmax(y_true_masked, axis=-1), K.argmax(y_pred_masked, axis=-1)))


def mcc_cc_loss(y_true_org, y_pred_org):
    mask = tf.reduce_any(tf.not_equal(y_true_org, MASK_VALUE), -1)
    y_true = tf.boolean_mask(y_true_org, mask)
    y_pred = tf.boolean_mask(y_pred_org, mask)

    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + K.epsilon()
    f1 = (tp * tn - fp * fn) / K.sqrt(denom)
    return K.mean(K.categorical_crossentropy(y_true, y_pred)) - K.mean(f1)


def filter_X(to_filter, true_list):
    return np.array([ss for ss, t in zip(to_filter, true_list) if t != "X"], dtype=object)


def generate_summary(protein_name, true_ss8, pred_ss8):
    reference_fasta = os.path.join(output_folder, f'{protein_name}_true.ss8')
    predicted_fasta = os.path.join(output_folder, f'{protein_name}_pred.ss8')
    if not os.path.isfile(reference_fasta) or not os.path.isfile(predicted_fasta):
        return
    metrics_file_path = os.path.join(output_folder, f'{protein_name}_metrics.txt')
    my_cmd = f"perl '{PATH_TO_SOV_SCRIPT}' '{reference_fasta}' '{predicted_fasta}' > '{metrics_file_path}'"
    subprocess.Popen(my_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).wait()
    os.remove(reference_fasta)
    os.remove(predicted_fasta)

    org_true = true_ss8.copy()
    true_ss8 = filter_X(true_ss8, org_true)
    pred_ss8 = filter_X(pred_ss8, org_true)

    with open(metrics_file_path, 'r') as file:
        lines = file.readlines()
        SOV_lines = [line for line in lines if line.startswith("SOV_refine")]
                    
    with open(metrics_file_path, "w") as metrics_file:
        cm = ConfusionMatrix(actual_vector=true_ss8, predict_vector=pred_ss8)
        AGMs = []
        F1s = []

        for line in SOV_lines:
            metrics_file.write(line)

        metrics_file.write("\n")
        for ss in cm.AGM:
            if cm.AGM.get(ss, 'None') != 'None':
                metrics_file.write(f"AGM for structure {ss}: {cm.AGM[ss]}\n")
                AGMs.append(cm.AGM[ss])
        metrics_file.write(f"Macro-avg AGM: {np.mean(AGMs)}\n")

        metrics_file.write("\n")
        for ss in cm.AGM:       
            true_ss8_ss = true_ss8[true_ss8 == ss]
            if len(true_ss8_ss) == 0:
                continue
            pred_ss8_ss = pred_ss8[true_ss8 == ss]

            correctness_list = np.equal(true_ss8_ss, pred_ss8_ss)
            
            try:
                q8 = np.mean(correctness_list)
            except:
                q8 = 0
            metrics_file.write(f"Q8 for structure {ss}: {q8}\n")

        correctness_list = np.equal(true_ss8, pred_ss8)
        metrics_file.write(f"Overall Q8: {np.mean(correctness_list)}\n")

        metrics_file.write("\n")
        for ss in cm.AGM:       
            one_hot_trues = np.where(true_ss8 == ss, 1, 0)
            one_hot_preds = np.where(pred_ss8 == ss, 1, 0)
            
            try:
                f1 = f1_score(one_hot_trues, one_hot_preds)
                metrics_file.write(f"F1 for structure {ss}: {f1}\n")
                F1s.append(f1)
            except Warning:
                pass
        metrics_file.write(f"Macro-avg F1: {np.mean(F1s)}\n")

        metrics_file.write("\n")
        metrics_file.write(f"MCC: {matthews_corrcoef(true_ss8, pred_ss8)}")

