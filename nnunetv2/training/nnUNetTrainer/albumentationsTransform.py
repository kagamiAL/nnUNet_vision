import torch
import albumentations as A
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class AlbumentationsTransform(BasicTransform):
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def get_parameters(self, **data_dict) -> dict:
        results = self.transforms(
            image=data_dict["image"].permute(1, 2, 0).numpy(),
            mask=data_dict["segmentation"][0].numpy(),
        )
        return {
            "new_image": results["image"],
            "new_segmentation": results["mask"],
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return torch.from_numpy(params["new_image"]).permute((2, 0, 1))

    def _apply_to_segmentation(
        self, segmentation: torch.Tensor, **params
    ) -> torch.Tensor:
        return torch.from_numpy(params["new_segmentation"]).unsqueeze(0)

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._apply_to_image(regression_target, **params)

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError
