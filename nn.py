import random
from engine import Value

class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def get_params(self):
        return self.w + [self.b]

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def __str__(self):
        return f"Neuron(w={[str(wi) for wi in self.w]}, b={str(self.b)})"

class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def get_params(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.get_params())
        return params

    def __call__(self, x) -> list[Value]:
        out = [n(x) for n in self.neurons]
        return out

class MLP:
    def __init__(self, nin, nouts):
        zs = [nin] + nouts
        self.layers = [Layer(zs[i], zs[i+1]) for i in range(len(zs) - 2)] + [Layer(zs[-2], zs[-1], nonlin=False)]

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def reset_param_grads(self):
        for param in self.get_params():
            param.grad = 0

    def __call__(self, x: list[float]) -> list[Value]:
        curr = [Value(xi) for xi in x]
        for layer in self.layers:
            curr = layer(curr)
        return curr



n = MLP(1, [10,5,1])

xs = [[1.0], [2.0], [3.0], [4.0]]

ys: list[list[float]] = [[1.0], [4.0], [9.0], [16.0]]

learning_rate = 0.001
num_iters = 1000

# TRAINING LOOP
for _ in range(num_iters):
    # forward pass, calculate loss
    ypreds = [n(xs[i]) for i in range(len(ys))]
    loss: Value = sum([(ypreds[j][i] - ys[j][i])**2 for i in range(len(ys[0])) for j in range(len(ys))], Value(0))

    # backpropagate, calculate gradients
    loss.backprop()

    # update parameters
    params = n.get_params()
    for param in params:
        param.data -= learning_rate * param.grad

    # reset parameter gradients
    n.reset_param_grads()

    # print running loss
    # print(loss.data)

print(loss.data)

