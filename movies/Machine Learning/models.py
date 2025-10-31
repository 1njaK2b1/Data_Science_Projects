import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        return self.w

    def run(self, x):
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        score = self.run(x)
        return 1 if nn.as_scalar(score) >= 0 else -1

    def train(self, dataset):

        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                score = self.run(x)
                pred = 1 if nn.as_scalar(score) >= 0 else -1
                label = nn.as_scalar(y)

                if pred != label:
                    x_data = x.data
                    update_data = label * x_data
                    update_node = nn.Constant(update_data)
                    self.w.update(update_node, 1.0)
                    converged = False


class RegressionModel(object):
    def __init__(self):
        self.l1 = nn.Parameter(1, 20)
        self.b1 = nn.Parameter(1, 20)
        self.l2 = nn.Parameter(20, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        x = nn.Linear(x, self.l1)        
        x = nn.AddBias(x, self.b1)         
        x = nn.ReLU(x)                   

        x = nn.Linear(x, self.l2)         
        x = nn.AddBias(x, self.b2)       
        return x

    def get_loss(self, x, y):
        prediction = self.run(x)
        return nn.SquareLoss(prediction, y)

    def train(self, dataset):
        loss = float('inf')
        while(loss>0.01):
            l1, b1, l2, b2 = nn.gradients(self.get_loss(nn.Constant(dataset.x),nn.Constant(dataset.y)), [self.l1, self.b1, self.l2, self.b2])
            self.l1.update(l1, -0.01)
            self.b1.update(b1, -0.01)
            self.l2.update(l2, -0.01)
            self.b2.update(b2, -0.01)
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x),nn.Constant(dataset.y)))

class DigitClassificationModel(object):
    def __init__(self):
        input_size = 784
        hidden_size = 200
        output_size = 10

        self.w1 = nn.Parameter(input_size, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)

        self.w2 = nn.Parameter(hidden_size, output_size)
        self.b2 = nn.Parameter(1, output_size)

    def run(self, x):
        h1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        h1_relu = nn.ReLU(h1)
        out = nn.AddBias(nn.Linear(h1_relu, self.w2), self.b2)
        return out

    def get_loss(self, x, y):
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        batch_size = 100
        learning_rate = 0.5
        params = [self.w1, self.b1, self.w2, self.b2]
        for epoch in range(20):
            for x_batch, y_batch in dataset.iterate_once(batch_size):
                loss = self.get_loss(x_batch, y_batch)
                grads = nn.gradients(loss, params)
                for i, param in enumerate(params):
                    param.update(grads[i], -learning_rate)

            val_acc = dataset.get_validation_accuracy()
            print(f"Epoch {epoch + 1}: val accuracy = {val_acc:.4f}")
            if val_acc >= 0.975:
                break

class LanguageIDModel(object):

    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        hidden_size = 128
        output_size = len(self.languages)

        self.Wx = nn.Parameter(self.num_chars, hidden_size)
        self.Wh = nn.Parameter(hidden_size, hidden_size)
        self.bh = nn.Parameter(1, hidden_size)

        self.W_output = nn.Parameter(hidden_size, output_size)
        self.b_output = nn.Parameter(1, output_size)

    def run(self, xs):
        batch_size = xs[0].data.shape[0]
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.Wx), self.bh))
        for i in range(1, len(xs)):
            x_proj = nn.Linear(xs[i], self.Wx)
            h_proj = nn.Linear(h, self.Wh)
            h = nn.ReLU(nn.AddBias(nn.Add(x_proj, h_proj), self.bh))
        return nn.AddBias(nn.Linear(h, self.W_output), self.b_output)

    def get_loss(self, xs, y):

        logits = self.run(xs)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):

        learning_rate = 0.2
        for epoch in range(100):
            for x_batch, y_batch in dataset.iterate_once(32):
                loss = self.get_loss(x_batch, y_batch)
                grads = nn.gradients(loss, [self.Wx, self.Wh, self.bh, self.W_output, self.b_output])
                self.Wx.update(grads[0], -learning_rate)
                self.Wh.update(grads[1], -learning_rate)
                self.bh.update(grads[2], -learning_rate)
                self.W_output.update(grads[3], -learning_rate)
                self.b_output.update(grads[4], -learning_rate)

            acc = dataset.get_validation_accuracy()
            print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}")
            if acc >= 0.87:
                break
