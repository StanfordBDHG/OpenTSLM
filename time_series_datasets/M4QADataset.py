from typing import List, Tuple, Literal, Optional
import pandas as pd
from pandas.io.sql import Series
import sys
import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QADataset import QADataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt


TEST_FRAC = 0.1
VAL_FRAC = 0.1

class M4QADataset(QADataset):
    def _load_splits(self) -> Tuple[List[dict], List[dict], List[dict]]:
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset_generation", "m4"
        )
        series_path = os.path.join(base_path, "m4_series_Weekly.csv")
        captions_path = os.path.join(base_path, "m4_captions_Weekly.csv")
        
        # Load the series and captions data
        series_df = pd.read_csv(series_path)
        captions_df = pd.read_csv(captions_path)
        
        # Merge the dataframes on the 'id' column
        df = pd.merge(series_df, captions_df, on='id')
        
        # Split the data into train, validation, and test sets
        df_train_val, df_test = train_test_split(
            df,
            test_size=TEST_FRAC,
            random_state=42,
            shuffle=True,
        )

        df_train, df_val = train_test_split(
            df_train_val,
            test_size=VAL_FRAC / (1.0 - TEST_FRAC),
            random_state=42,
            shuffle=True,
        )

        return (
            df_train.to_dict(orient="records"),
            df_val.to_dict(orient="records"),
            df_test.to_dict(orient="records"),
        )

    def _get_answer(self, row) -> str:
        return row["Caption"]

    def _get_pre_prompt(self, row) -> str:
        return "Given this time series data:"

    def _get_post_prompt(self, row) -> str:
        return "Generate a detailed description."

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        try:
            series_list = json.loads(row["Series"])
            series_array = np.array(series_list, dtype=np.float32)
            
            prompt = TextTimeSeriesPrompt(
                "This is the time series.",
                time_series=series_array
            )
            return [prompt]
        except Exception as e:
            print(f"Error parsing series data: {e}")
            print(f"Series data type: {type(row['Series'])}")
            print(f"Series data: {row['Series'][:100]}...")
            raise

if __name__ == "__main__":
    train = M4QADataset("train", "")
    print(train[0])

