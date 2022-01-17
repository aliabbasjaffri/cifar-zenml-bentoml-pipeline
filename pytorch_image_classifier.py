from typing import List, BinaryIO

from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from datasource import classes
import bentoml
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import FileInput


@bentoml.env(pip_packages=["torch", "numpy", "torchvision", "scikit-learn"])
@bentoml.artifacts([PytorchModelArtifact("net")])
class PytorchImageClassifier(bentoml.BentoService):
    @bentoml.utils.cached_property
    def transform(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        input_datas = []
        for fs in file_streams:
            img = Image.open(fs).resize((32, 32))
            input_datas.append(self.transform(img))

        outputs = self.artifacts.net(Variable(torch.stack(input_datas)))
        _, output_classes = outputs.max(dim=1)

        return [classes[output_class] for output_class in output_classes]


def save_classifier(net):
    # 2) `pack` it with required artifacts
    bento_svc = PytorchImageClassifier()
    bento_svc.pack("net", net)

    # 3) save your BentoSerivce to file archive
    saved_path = bento_svc.save()
    print(saved_path)
