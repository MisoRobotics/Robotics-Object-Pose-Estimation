"""Load data captured with a Unity Perception Camera."""
import logging
import os
import zipfile
from typing import (
    Final,
    Tuple,
)

import torch
import yaml
from datasetinsights.datasets.unity_perception import Captures
from datasetinsights.io.downloader import GCSDatasetDownloader
from easydict import EasyDict
from pandas import DataFrame
from torch import Tensor
from torchvision.io import (
    ImageReadMode,
    read_image,
)
from torchvision.transforms import Resize

_config_path = os.path.join(os.path.dirname(__file__), "..", "dataset.yaml")

logger = logging.getLogger(__name__)


def _download_dataset(source_uri: str, dest: str) -> None:
    downloader = GCSDatasetDownloader()
    downloader.download(source_uri=source_uri, output=dest)
    zip_file = os.path.join(dest, os.path.basename(source_uri))
    logging.info("Extracting dataset to: %s", dest)
    with zipfile.ZipFile(zip_file, "r") as stream:
        stream.extractall(dest)


class UnityPerceptionDataset(torch.utils.data.Dataset):
    """Load data from Unity Perception cameras."""

    def __init__(self, config: EasyDict, split_name: str) -> None:
        """Construct a new UnityPerceptionDataset.

        Args:
            config: Configuration about the dataset to load.
            split: Subpath under the data root for this set.

        """
        root = config.system.data_root
        self._config: Final[EasyDict] = config
        self._data_root: Final[str] = os.path.join(root, split_name)

        if not config.dataset.skip_download:
            _download_dataset(config[split_name].source_uri, self._data_root)
        else:
            logger.warning("Skipping download due to configuration setting.")

        captures = Captures(data_root=self._data_root)
        annotation_id = config.dataset.annotation_id
        self._data: Final[DataFrame] = captures.filter(def_id=annotation_id)
        logging.info("Loaded dataset with %d images.", len(self))

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the image and labels for the specified index.

        Args:
            idx: The index for the dataset. Zero is the first index.

        Returns:
            The image and label corresponding to the specified index.

        """
        path = os.path.join(self._data_root, self._data["filename"][idx])
        mode = ImageReadMode.RGB
        image = read_image(path, mode)
        scale = self._config.dataset.image_scale
        transform = Resize((scale, scale))
        scaled_image = transform(image).unsqueeze(0)

        ego = self._data["ego"][idx]
        translation = torch.tensor(ego["translation"], dtype=torch.float)
        rotation = torch.tensor(ego["rotation"], dtype=torch.float)

        return scaled_image, translation, rotation

    @classmethod
    def from_default_config(cls, split: str):
        """Construct a UnityPerceptionDataset from defaults."""
        with open(_config_path, "r") as stream:
            config = EasyDict(yaml.load(stream, Loader=yaml.FullLoader))

        logger.info("Loaded config for dataset '%s':\n\n%s", split, config)
        return cls(config, split)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=("%(levelname)s | %(asctime)s | %(name)s | %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    UnityPerceptionDataset.from_default_config("train")
