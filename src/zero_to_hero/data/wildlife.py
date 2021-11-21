"""
Dataset & DataModule for Blobs
"""
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import imageio


class UnpackingError(Exception):
    """
    Customized Exception Class for failures during unpackaging of objects
    """

    def __init__(self, fault_object: Any, message: str = "Unpacking of object failed.") -> None:
        self.message = f"{message} Length of object: {len(fault_object)}."
        super().__init__(message)


class WildlifeDataset(Dataset):
    """
    Implementation of the Blobs' PyTorch Dataset
    """

    def __init__(self, data: np.ndarray, targets: Optional[np.ndarray] = None) -> None:
        """
        Initialize Wildlife Dataset
        :param data: np.ndarray with shape (n_samples, n_features)
        :param targets: np.ndarray with shape (n_samples, 1)
        """
        self.data = data
        self.size = targets.shape[0]
        if targets is not None:
            assert self.size == targets.shape[0], "Number of input data is not equal to the number of targets"
        self.targets = targets

    def __len__(self) -> int:
        """

        :return: n_samples
        """
        return self.size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param idx: int received from sampler
        :return: (datum, target)
        """
        if self.targets is not None:
            data = np.array(imageio.imread(self.data[idx])[:,:, 0].astype(np.float32))
            data = data.reshape((1, data.shape[0], data.shape[1]))
            return data, self.targets[idx]
            # return self.data[idx], self.targets[idx]
        return self.data[idx]


class WildlifeDataModule(pl.LightningDataModule):
    """
    Implementation of the Blobs' PyTorch Lightning DataModule
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.config = config

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        assert stage in (None, "fit", "validate", "test", "predict")

        print("Training data & targets.")
        train_data, train_targets = self.get_data_and_targets(Path(self.config["data"]["train_data_dir"]))
        print("Validation data & targets.")
        valid_data, valid_targets = self.get_data_and_targets(Path(self.config["data"]["valid_data_dir"]))
        print("Testing data & targets.")
        test_data, test_targets = self.get_data_and_targets(Path(self.config["data"]["test_data_dir"]))
        # print("Predicting data & targets.")
        # predict_data, _ = self.load_frames(predict_data_dir)

        self.train_dataset = WildlifeDataset(
            data=train_data,
            targets=train_targets,
        )
        self.valid_dataset = WildlifeDataset(
            data=valid_data,
            targets=valid_targets,
        )
        self.test_dataset = WildlifeDataset(
            data=test_data,
            targets=test_targets,
        )
        # self.predict_dataset = WildlifeDataset(
        #     data=predict_data,
        #     targets=None,
        # )

    def load_frames(self, frame_dir: Path) -> Dict[int, Dict[str, Union[bool, np.ndarray]]]:
        """
        Loads frame files from a directory.
        :param frame_dir: Path to directory where frames are stored in either a label or no_label folder
        :return: Dictionary where key is frame index and value is a dictionary with the label class and frame image
        """

        frame_filepaths = list(frame_dir.rglob("*.jpeg"))

        frame_dict = {}

        for ind, frame_filepath in enumerate(frame_filepaths):

            # frame_img = imageio.imread(frame_filepath)

            frame_index = ind # int(frame_filepath.stem.split("___")[-1])

            if frame_filepath.parent.stem not in ["label", "no_label"]:
                raise FileNotFoundError("File structure not recognized. frames should be "
                                        "placed in 'label' and 'no_label' folders")

            if frame_filepath.parent.stem == "label":
                label = True
            else:
                label = False

            frame_dict[frame_index] = {"path": frame_filepath,
                                       "label": label}
        return frame_dict

    def get_data_and_targets(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the feature data and targets (blobs) for each given center.
        :return: (features, target)
        """
        frame_dicts = self.load_frames(data_path)
        # data_list = [frame_dict["img"][:, :, 0] for _, frame_dict in frame_dicts.items()]
        data_list = [frame_dict["path"] for _, frame_dict in frame_dicts.items()]
        target_list = [frame_dict["label"] for _, frame_dict in frame_dicts.items()]
        #data = np.stack(data_list, axis=-1).transpose()
        #data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2])).astype(np.float32)
        targets = np.hstack(target_list).astype(np.int64)
        data = data_list
        return data, targets

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert isinstance(self.train_dataset, Dataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=True,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.valid_dataset, Dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.test_dataset, Dataset)
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        assert isinstance(self.predict_dataset, Dataset)
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.config["hyper_parameters"]["batch_size"],
            shuffle=False,
            pin_memory=self.config["hyper_parameters"]["pin_memory"],
            num_workers=self.config["hyper_parameters"]["num_workers"],
            drop_last=False,
        )
