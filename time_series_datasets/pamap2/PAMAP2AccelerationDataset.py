import pandas as pd
from time_series_datasets.pamap2.PAMAP2Dataset import PAMAP2Dataset


class PAMAP2AccelerationDataset(PAMAP2Dataset):
    """
    A subclass of PAMAP2Dataset that only includes acceleration data.
    """

    def _data_cleaning(self, dataCollection):
        dataCollection = super()._data_cleaning(dataCollection)
        
        all_columns = dataCollection.columns.tolist()
        
        columns_to_keep = []
        for col in all_columns:
            if col in ['timestamp', 'activityID', 'subject_id']:
                columns_to_keep.append(col)
            elif 'Acc' in col:
                columns_to_keep.append(col)
        
        return dataCollection[columns_to_keep]
