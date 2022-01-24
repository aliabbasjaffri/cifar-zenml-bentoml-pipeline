from datasource import get_trainset, get_testset
import torch
from torch.nn import Module
from pytorch_image_classifier import PytorchImageClassifier
from trainer import cifar_trainer
from zenml.pipelines import pipeline
from zenml.steps import step


@step
def trainer() -> Module:
    """
    A simple pytorch Model to train on the data.
    """
    train = get_trainset()
    network = cifar_trainer(train)
    return network


@step
def evaluator(
    net: Module
) -> float:
    """
    Calculate the accuracy on the test set
    """
    correct = 0
    total = 0
    test_loader = get_testset()
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )

    return 100 * correct / total


@step
def save_model(net: Module):
    """
    Saves model in BentoML format
    :param net: The model that is learned during the training
    """
    bento_svc = PytorchImageClassifier()
    bento_svc.pack("net", net)

    # 3) save your BentoService to file archive
    saved_path = bento_svc.save()
    print(saved_path)


@pipeline
def cifar_pipeline(
    _trainer,
    _evaluator,
    _save_model,
):
    """Links all the steps together in a pipeline"""

    net = _trainer()
    _evaluator(net)
    _save_model(net)


if __name__ == "__main__":
    # Run the pipeline
    p = cifar_pipeline(
        _trainer=trainer(),
        _evaluator=evaluator(),
        _save_model=save_model()
    )
    p.run()
