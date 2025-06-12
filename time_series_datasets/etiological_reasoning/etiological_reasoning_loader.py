
from constants import RAW_DATA_PATH
import os
import subprocess



def load_etiological_reasoning_dataset():

    # Load the PTB-XL dataset
    etiological_reasoning_dir = os.path.join(RAW_DATA_PATH, "etiological_reasoning")
    if not os.path.exists(etiological_reasoning_dir):
        # @Winnie here you could add code to download the dataset automatically
        # and save it to the etiological_reasoning_dir.
        raise FileNotFoundError(f"etiological_reasoning directory not found at {etiological_reasoning_dir}")

    # Return the raw data, e.g. a pandas dataframe.
    # Whatever you need to work with it in the EtiologicalReasoningDatasetLoader.



    return mimic_iv_ecg_dir
if __name__ == "__main__":
    data = load_etiological_reasoning_dataset()

