import torch.nn as nn
from torch import optim


def get_loss_func(loss_string):
    """
    Helper function to get the loss function given the name of the loss function
    :param loss_string: the name of the loss function
    :return: the pytorch loss function, whether the output is categorical or numerical
    """
    if not isinstance(loss_string, str):
        if len(loss_string) != 2:
            raise TypeError("Customized loss layer need to return whether output is categorical")
        return loss_string

    loss_string = loss_string.lower()
    criterion_ls = {"crossentropy": nn.NLLLoss(), "mse": nn.MSELoss(), "binarycrossentropy": nn.BCELoss()}
    out_put_categorical = {"crossentropy": True, "mse": False, "binarycrossentropy": True}
    return criterion_ls[loss_string], out_put_categorical[loss_string]


def get_accuracy(loss_string, output, targets, batch_size, threshold=0.5):
    """
    Helper function to calculate the accuracy of a given batch
    :param loss_string: loss function currently used
    :param output: the model prediction
    :param targets: target output
    :param batch_size: batch size
    :param threshold: decision threshold for binary classification
    :return: the accuracy of current batch
    """
    if loss_string == "crossentropy":
        max_index = output.max(dim=1)[1]
        correct = (max_index == targets).sum()
        accuracy = int(correct.data) / len(targets)

    elif loss_string == "binarycrossentropy":
        binary_output = output > threshold
        correct = sum((binary_output.float() == targets).data.cpu().numpy()[0])
        accuracy = correct / batch_size

    else:
        raise ValueError("Accuracy metrics not supported to current network")

    return accuracy


def get_optimizer(optimizer_string, parameters):
    """
    Helper function to get optimizer from the optimizer name
    :param optimizer_string: name of the optimizer
    :param parameters: the model parameters
    :return: pytorch optimizer with model parameters to optimize
    """
    if not isinstance(optimizer_string, str):
        return optimizer_string
    else:
        opt_ls = {"adam": optim.Adam(parameters, lr=1e-3)}
        return opt_ls[optimizer_string]
