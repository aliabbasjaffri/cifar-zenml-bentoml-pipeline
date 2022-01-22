from typing import Any, Type
from network import Net
import torch
import os
from zenml.artifacts import DataArtifact
from zenml.materializers.base_materializer import BaseMaterializer


DEFAULT_FILENAME = "model.pt"


class NetworkMaterializer(BaseMaterializer):
    """Materializer to read/write Network."""

    ASSOCIATED_TYPES = [Net]
    ASSOCIATED_ARTIFACT_TYPES = [DataArtifact]

    def handle_input(self, data_type: Type[Any]) -> Net:
        """Reads and returns a PyTorch network.

        Returns:
            A loaded pytorch model.
        """
        super().handle_input(data_type)
        filename = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        print(filename)
        model = torch.load_state_dict(torch.load(filename))
        return model

    def handle_return(self, net: Net) -> None:
        """Writes a PyTorch DataLoader.

        Args:
            net: A torch.utils.data.DataLoader or a dict to pass into model.save
        """
        super().handle_return(net)
        filename = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        print(filename)
        torch.save(net.state_dict(), filename)


