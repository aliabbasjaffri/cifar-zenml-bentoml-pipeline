from datasource import get_dataset
import torch
from network import Net
from trainer import CIFAR_trainer
from pytorch_image_classifier import save_classifier

from zenml.pipelines import pipeline
from zenml.steps import step, Output


@step
def importer() -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    Download the CIFAR data.
    """
    train, test = get_dataset()
    return train, test


@step
def trainer(
    train_loader: torch.utils.data.DataLoader
) -> Net:
    """A simple pytorch Model to train on the data."""
    net = CIFAR_trainer(train_loader)
    return net


@step
def evaluator(
    net: Net,
    test_loader: torch.utils.data.DataLoader
) -> float:
    """Calculate the accuracy on the test set"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
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
    save_classifier(net)


@pipeline
def cifar_pipeline(
    importer,
    trainer,
    evaluator,
    save_model,
):
    """Links all the steps together in a pipeline"""
    train, test = importer()
    model = trainer(train)
    evaluator(model, test)
    save_model(model)


if __name__ == "__main__":
    # Run the pipeline
    p = cifar_pipeline(
        importer=importer(),
        trainer=trainer(),
        evaluator=evaluator(),
        save_model=save_model()
    )
    p.run()
