from typing import Any, Type
from torch.utils.data import DataLoader
from zenml.artifacts import DataArtifact
from zenml.materializers.base_materializer import BaseMaterializer
import pickle
import os
from zenml.io import fileio

DEFAULT_FILENAME = 'data.pickle'


class DataLoaderMaterializer(BaseMaterializer):
    """Materializer to read/write Pytorch data loaders."""

    ASSOCIATED_TYPES = [DataLoader]
    ASSOCIATED_ARTIFACT_TYPES = [DataArtifact]

    def handle_input(self, data_type: Type[Any]) -> DataLoader:
        """Reads and returns a PyTorch DataLoader.

        Returns:
            A loaded pytorch model.
        """
        super().handle_input(data_type)
        # read from the directory self.artifact.uri and return dataloader
        filename = os.path.join(self.artifact.uri, DEFAULT_FILENAME)

        with fileio.open(filename, 'rb') as f:
            return pickle.load(f)

    def handle_return(self, dataloader: DataLoader) -> None:
        """Writes a PyTorch DataLoader.

        Args:
            dataloader: A torch.utils.data.DataLoader or a dict to pass into model.save
        """
        super().handle_return(dataloader)
        # write to the directory self.artifact.uri
        filename = os.path.join(self.artifact.uri, DEFAULT_FILENAME)

        with fileio.open(filename, 'wb') as f:
            pickle.dump(dataloader, f)
