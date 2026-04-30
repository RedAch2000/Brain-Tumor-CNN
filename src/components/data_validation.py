from dataclasses import dataclass
from pathlib import Path
# from src.utils.download_kaggle import count_images_per_class


@dataclass
class DataValidationConfig:
    data_dir: Path = Path("dataset/raw")


class DataValidationStep:

    def __init__(self):
        self.config = DataValidationConfig()

    def run(self):
        try:
            data_dir = self.config.data_dir

            if not data_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

            # Get class folders
            classes = [d.name for d in data_dir.iterdir() if d.is_dir()]

            #  verifying that we have only two classes (yes and no)
            if len(classes) != 2:
                raise ValueError(
                    f"Expected exactly 2 classes, found {len(classes)}: {classes}"
                )

            # count images per class
            # counts = count_images_per_class(data_dir)
            # print("\n Images per class (after download):")
            # for cls, n in counts.items():
            #     print(f"{cls}: {n}")

            print("Data Validation Completed Successfully.")

            return {
                "validated_data_dir": data_dir,
                "classes": classes,
                # "class_counts": counts,
                # class_weights = [w_no, w_yes] 
                # with  w_no = total_samples / (number_of_classes * n_class_no) 
                # and w_yes = total_samples / (number_of_classes * n_class_yes)
                # they are used to handle class imbalance during training
                # "class_weights": [
                #     (counts["no"] + counts["yes"])/(len(classes) * counts["no"]), 
                #     (counts["no"] + counts["yes"])/(len(classes) * counts["yes"])
                # ]
            }

        except Exception as e:
            raise RuntimeError(f"Error in Data Validation Step: {e}")