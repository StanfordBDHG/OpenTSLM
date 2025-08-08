
from constants import RAW_DATA_PATH
import os
import subprocess

def does_ecg_qa_exist():
    return os.path.exists(os.path.join(RAW_DATA_PATH, "ecg_qa"))

def clone_ecg_qa():
    print(f"Cloning ECG-QA into ./{RAW_DATA_PATH}/ecg_qa …")
    subprocess.run(["git", "clone", "https://github.com/Jwoo5/ecg-qa", RAW_DATA_PATH], check=True)

def download_ecg_qa_if_not_exists():
    if not does_ecg_qa_exist():
        clone_ecg_qa()

def load_ecg_qa_ptbxl():

    download_ecg_qa_if_not_exists()
    data_dir = os.path.join(RAW_DATA_PATH, "ecg_qa")
    # Load the PTB-XL dataset
    if not os.path.exists(data_dir):
        print(f"Cloning ECG-QA into ./{data_dir} …")
        subprocess.run(["git", "clone", "https://github.com/Jwoo5/ecg-qa", data_dir], check=True)

    ptbxl_dir = os.path.join(data_dir, "ptbxl")
    if not os.path.exists(ptbxl_dir):
        raise FileNotFoundError(f"PTB-XL directory not found at {ptbxl_dir}")

    return ptbxl_dir

def load_ecg_qa_mimic_iv_ecg():
    download_ecg_qa_if_not_exists()
    data_dir = os.path.join(RAW_DATA_PATH, "ecg_qa")
    mimic_iv_ecg_dir = os.path.join(data_dir, "mimic-iv-ecg")
    if not os.path.exists(mimic_iv_ecg_dir):
        raise FileNotFoundError(f"MIMIC-IV-ECG directory not found at {mimic_iv_ecg_dir}")

    return mimic_iv_ecg_dir
if __name__ == "__main__":
    load_ecg_qa_ptbxl()

