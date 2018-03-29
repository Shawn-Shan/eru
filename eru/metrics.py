import torch.nn as nn
from torch import optim


def get_loss_func(loss_string):
    
    if not isinstance(loss_string, str):
        if len(loss_string) != 2:
            raise TypeError("Customized loss layer need to return whether output is categorical")
        return loss_string
        
    loss_string = loss_string.lower()
    criterion_ls = {"crossentropy": nn.NLLLoss(), "mse": nn.MSELoss(), "binarycrossentropy": nn.BCELoss()}
    out_put_categorical = {"crossentropy": True, "mse": False, "binarycrossentropy": True}
    return criterion_ls[loss_string], out_put_categorical[loss_string]


def get_accuracy(loss_string, output, targets, batch_size, threshold=0.5):
    if loss_string == "crossentropy":
        max_index = output.max(dim=1)[1]
        correct = (max_index == targets).sum()
        accuracy = int(correct.data) / len(targets)

    elif loss_string == "binarycrossentropy":
        binary_output = output > threshold
        correct = sum((binary_output.float() == targets).data.cpu().numpy()[0])
        accuracy = correct / batch_size

    return accuracy


def get_optimizer(optimizer_string, parameters):
    if not isinstance(optimizer_string, str):
        return optimizer_string
    else:
        opt_ls = {"adam": optim.Adam(parameters, lr=1e-3)}
        return opt_ls[optimizer_string]
