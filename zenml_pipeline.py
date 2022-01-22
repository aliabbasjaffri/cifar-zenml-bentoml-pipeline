from NetworkMaterializer import NetworkMaterializer
from datasource import get_trainset, get_testset
import torch
from network import Net
from trainer import cifar_trainer
from torch.utils.data import DataLoader

from zenml.pipelines import pipeline
from zenml.steps import step, Output


# @step
def importer() -> (DataLoader, DataLoader):
    """
    Download the CIFAR data.
    """
    train = get_trainset()
    test = get_testset()
    return train, test


@step
def trainer() -> Output(network=Net):
    """A simple pytorch Model to train on the data."""
    # train, test = importer()
    train = get_trainset()
    network = cifar_trainer(train)
    return network


@step
def evaluator(
    net: Net
) -> float:
    """Calculate the accuracy on the test set"""
    correct = 0
    total = 0
    test_loader = get_testset()
    print(test_loader)
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
def save_model(net: Net) -> None:
    """
    Saves model in BentoML format
    :param net: The model that is learned during the training
    :return: Nothing
    """
    # save_classifier(net)


@pipeline
def cifar_pipeline(
    # _importer,
    _trainer,
    _evaluator,
    # _save_model,
):
    """Links all the steps together in a pipeline"""

    network = _trainer()
    _evaluator(network)
    # _save_model(net)


if __name__ == "__main__":
    # Run the pipeline
    p = cifar_pipeline(
        # _importer=importer(),
        _trainer=trainer().with_return_materializers(NetworkMaterializer),
        _evaluator=evaluator()
        # _evaluator=evaluator().with_return_materializers(DataLoaderMaterializer),
        # _save_model=save_model()
    )
    p.run()
