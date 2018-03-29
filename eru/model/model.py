from __future__ import absolute_import


class Model(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(Model, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.layers = []
        self.params = []
        self.traverse_layers(input_layer)
        
        
    def forward_layer(self, input, cur_layer):
        output = cur_layer.forward(input)
        if not cur_layer.out_bound_layers:
            return output
        
        for layer in cur_layer.out_bound_layers:
            return self.forward_layer(output, layer)
    
    def parameters(self):
        return self.params
    
    def traverse_layers(self, cur_layer):
        for layer in cur_layer.out_bound_layers:
            self.layers.append(layer)
            self.params += list(layer.parameters())
            self.traverse_layers(layer)
            
    def change_batch_size(self, bsz):
        self.batch_size = batch_size

        for layer in self.layers:
            layer.cuda()
            layer.update_batch_size(bsz)
            
            
    def forward(self, input):
        return self.forward_layer(input, self.input_layer)
    
    
    def compile(self, optimizer="adam", criterion="crossentropy", clip_norm=True, metrics=[]):

        optimizer = get_optimizer(optimizer, self.parameters())
        loss_func, self.output_categorical = get_loss_func(criterion)
        self.criterion_string = criterion
        self.complied = True
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.clip_norm = clip_norm
        self.metrics = metrics

    def train_one_epoch(self):
        if not self.complied:
            raise TypeError("Model need to be compiled before training")

        self.train()
        for batch, i in enumerate(range(0, self.training_length)):
            data, targets = next(self.data_generator)
            self.zero_grad()
            self.optimizer.zero_grad()
            output = self(data)
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
        self.change_batch_size(batch_size)

        if train_length is None:
            self.training_length = len(data_loader)
        else:
            self.training_length = train_length

        self.data_loader = data_loader
        self.data_generator = data_loader.generate()
        self.train_run(epochs=epochs)

    def train_run(self, epochs=1):
        try:
            for epoch in range(1, epochs + 1):
                self.progress_bar = Progbar(self.training_length)
                epoch_start_time = time.time()
                self.train_one_epoch()
                delta_time = time.time() - epoch_start_time
                print("\n")

        except KeyboardInterrupt:
            print("\n")
            print('-' * 89)
            print('Exiting from training early')

    def evaluation(self, test_loader, batch_size=1):
        self.change_batch_size(batch_size)
        self.eval()

        text_generator = test_loader.generate()
        test_length = len(test_loader)
        test_progress_bar = Progbar(test_length)
        total_loss = 0
        total_acc = 0
        total_preplexity = 0

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

        return total_loss / test_length, total_acc / test_length, total_preplexity / test_length


    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self, f)
    
    