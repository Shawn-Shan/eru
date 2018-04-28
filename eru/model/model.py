from __future__ import absolute_import
from ..metrics import *
from ..train import *
from ..utils import *


class Model(nn.Module):
    """The `Model` class adds training & evaluation routines to a `Network`.
    """

    def __init__(self, input_layer, output_layer):
        """
        Initialize model and build model architecture

        :param input_layer: an input layer instant
        :param output_layer: an final layer instant
        """
        super(Model, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers = []
        self.params = []
        self.traverse_layers(input_layer)

        self.criterion_string = None
        self.complied = False
        self.optimizer = None
        self.loss_func = None
        self.clip_norm = None
        self.metrics = None
        self.output_categorical = None
        self.batch_size = None
        self.data_loader = None
        self.training_length = None
        self.progress_bar = None

    def forward_layer(self, input, cur_layer):
        """
        Feedforward a layer
        :param input: input from connected layers
        :param cur_layer: current layer
        :return: output of current layer
        """
        output = cur_layer.forward(input)
        if not cur_layer.out_bound_layers:
            return output

        for layer in cur_layer.out_bound_layers:
            return self.forward_layer(output, layer)

    def parameters(self):
        """
        Recompile current model parameters for optimizer
        :return: all model parameters
        """
        return self.params

    def traverse_layers(self, cur_layer):
        """
        Build the model architecture graph
        """
        for layer in cur_layer.out_bound_layers:
            self.layers.append(layer)
            self.params += list(layer.parameters())
            self.traverse_layers(layer)

    def change_batch_size(self, bsz):
        """
        Change model batch size
        :param bsz: target batch size
        """
        self.batch_size = bsz

        for layer in self.layers:
            layer.update_batch_size(bsz)

    def forward(self, input):
        """
        Main forward function of the entire model
        :param input: input data
        :return: output data
        """
        if self.output_categorical:
            return self.forward_layer(input, self.input_layer).view(self.batch_size, -1)
        else:
            return self.forward_layer(input, self.input_layer).view(-1, self.batch_size)

    def compile(self, optimizer="adam", criterion="crossentropy", clip_norm=True, metrics=[]):
        """
        Compile the model with training configuration. Model must be complied before training
        :param optimizer: optimizer used for training
        :param criterion: loss function for training
        :param clip_norm: whether to use clip_norm
        :param metrics: evaluation metrics
        """

        optimizer = get_optimizer(optimizer, self.parameters())
        loss_func, self.output_categorical = get_loss_func(criterion)

        self.criterion_string = criterion
        self.complied = True
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.clip_norm = clip_norm
        self.metrics = metrics

    def train_one_epoch(self, generator):
        """
        Train the model for one epochs
        :param generator: a data generator that yield [x, y] batch at every step
        """
        if not self.complied:
            raise TypeError("Model need to be compiled before training")

        self.train()
        for batch, i in enumerate(range(0, self.training_length)):
            data, targets = next(generator)
            self.zero_grad()
            self.optimizer.zero_grad()
            output = self.__call__(data)
            loss = self.loss_func(output, targets)
            loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm(self.parameters(), 0.25)

            self.optimizer.step()
            report = []

            if i % 20 == 0:
                if "loss" in self.metrics:
                    report.append(("loss", loss.data[0]))
                if "acc" in self.metrics:
                    accuracy = get_accuracy(self.criterion_string, output, targets, self.batch_size)
                    report.append(("acc", accuracy))
                if "perplexity" in self.metrics:
                    report.append(("perplexity", torch.exp(loss).data[0]))

                self.progress_bar.update(i, report)

    def fit_generator(self, data_loader, batch_size, epochs, train_length=None):
        """
        Training API and train the data with a data generator which yield [x, y] batch at every step
        :param data_loader: the data generator
        :param batch_size: training batch_size
        :param epochs: number of epochs
        :param train_length: the number of batches of training
        """
        self.change_batch_size(batch_size)

        if train_length is None:
            self.training_length = len(data_loader)
        else:
            self.training_length = train_length

        self.data_loader = data_loader
        self.train_run(epochs=epochs, generator=data_loader.generate())

    def fit(self, x_train, y_train, x_test=None, y_test=None, batch_size=32, epochs=1, shuffle=True):
        """
        Training API take two lists of x and y
        :param x_train: training input data
        :param y_train: target output data
        :param x_test: testing input data
        :param y_test: testing target data
        :param batch_size: training batch_size
        :param epochs: number of epochs
        :param shuffle: shuffle the training or not
        """
        self.change_batch_size(batch_size)
        self.data_loader = simple_loader(x_train, y_train, x_test, y_test, batch_size=batch_size, shuffle=shuffle)

        self.training_length = len(self.data_loader)
        self.train_run(self.data_loader, epochs=epochs)

    def train_run(self, generator, epochs=1):
        """
        Wrapper to train the model for multiple epochs
        :param generator:
        :param epochs:
        :return:
        """
        try:
            for epoch in range(1, epochs + 1):
                self.progress_bar = Progbar(self.training_length)
                self.train_one_epoch(generator)
                print("\n")

        except KeyboardInterrupt:
            print("\n")
            print('-' * 89)
            print('Exiting from training early')

    def evaluation(self, test_loader, batch_size=1):
        """
        Evalution on test data from a test data generator
        :param test_loader: the testing data generator
        :param batch_size: testing batch size
        :return: metrics on the testing data
        """
        self.change_batch_size(batch_size)
        self.eval()

        text_generator = test_loader.generate()
        test_length = len(test_loader)
        test_progress_bar = Progbar(test_length)
        total_loss = 0
        total_acc = 0
        total_preplexity = 0
        return_metrics = []
        for i in range(0, test_length):
            data, targets = next(text_generator)
            output = self(data)
            loss = self.loss_func(output, targets)
            total_loss += loss.data[0]
            report = []
            if "loss" in self.metrics:
                report.append(("loss", loss.data[0]))
                total_loss += loss.data[0]
            if "acc" in self.metrics:
                accuracy = get_accuracy(self.criterion_string, output, targets, self.batch_size)
                report.append(("acc", accuracy))
                total_acc += accuracy
            if "perplexity" in self.metrics:
                report.append(("perplexity", torch.exp(loss).data[0]))
                total_preplexity += total_preplexity.data[0]

            if i % 20 == 0:
                test_progress_bar.update(i, report)

        if "loss" in self.metrics:
            return_metrics.append(total_loss / test_length)
        if "acc" in self.metrics:
            return_metrics.append(total_acc / test_length)

        if "perplexity" in self.metrics:
            return_metrics.append(total_preplexity / test_length)

        return return_metrics

    def save(self, file_name):
        """
        Save the entire model class to a local file
        :param file_name: the save path
        """
        with open(file_name, 'wb') as f:
            torch.save(self, f)
