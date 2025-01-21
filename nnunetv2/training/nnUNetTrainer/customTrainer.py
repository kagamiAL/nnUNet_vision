import torch
import os
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join


class CustomTrainer(nnUNetTrainer):
    data_augmentation_random_crop: bool = False

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.output_folder = join(
            self.output_folder,
            os.environ.get("model_name", ""),
        )
        self.num_epochs = 700
