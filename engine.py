class Value():
    def __init__(self, data, grad=0.0, op='', partial=lambda: None, children=()):
        self.data = data
        self.grad = grad
        self.backward = lambda: None
        self.op = op
        self.partial = partial
        self._prev = children

    def backprop(self):
        self.grad = 1.0

        def topo_sort(root):
            visited = set()
            order = []

            def dfs(node):
                if node in visited:
                    return
                visited.add(node)

                for child in node._prev:
                    dfs(child)

                order.append(node)

            dfs(root)
            return order

        order = reversed(topo_sort(self))
        for value in order:
            value.backward()

    def relu(self):
        out = Value(self.data if self.data >= 0 else 0, children=(self, ), op='ReLU')

        def backward():
            self.grad += out.grad * (out.grad > 0)

        out.backward = backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op='+')

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            # print(out.data, out.grad, self.grad, other.grad)

        out.backward = backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op='*')

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            # print(out.data, out.grad, self.grad, other.grad)

        out.backward = backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(-1.0*other)

    def __rsub__(self, other):
        return (self * -1.0).__add__(other)

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, children=(self, other), op='**')

        def backward():
            self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
            # other.grad is useless to update

        out.backward = backward
        return out

    def __str__(self):
        return f"Value({self.data, self.op, [c.data for c in self._prev]})"

# a = Value(0)
# b = Value(5)
# c = a * b
# d = b + a
# e = c + d
# f = e.relu()
# print(f)
# f.backprop()

# print(a.grad)
# print(b.grad)