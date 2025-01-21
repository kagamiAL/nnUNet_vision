from nnunetv2.training.nnUNetTrainer.customTrainer import CustomTrainer


class RandomCropTrainer(CustomTrainer):
    data_augmentation_random_crop: bool = True
