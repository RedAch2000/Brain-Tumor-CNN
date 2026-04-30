from dataclasses import dataclass
from src.utils.download_kaggle import download_kaggle_dataset
from pathlib import Path
import os

@dataclass
class DataIngestionConfig:
    data_dir_raw: Path = Path("dataset/raw")
    path_kaggle_dataset: str = "navoneel/brain-mri-images-for-brain-tumor-detection"


class DataIngestionStep:

    def __init__(self):
        self.config = DataIngestionConfig()

    def run(self):
        try:
            download_kaggle_dataset(
                self.config.path_kaggle_dataset,
                self.config.data_dir_raw
            )
            print("Data Ingestion Completed Successfully.")
            return (
                self.config.data_dir_raw
            )

        except Exception as e:
            raise Exception(f"Error in Data Ingestion Step: {str(e)}")

