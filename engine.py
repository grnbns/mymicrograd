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
        # print([str(o) for o in order])
        for value in order:
            value.backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op='+', partial=1.0)

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            print(out.data, out.grad, self.grad, other.grad)

        out.backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op='*', partial=other)

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            print(out.data, out.grad, self.grad, other.grad)

        out.backward = backward
        return out

    def __str__(self):
        return f"Value({self.data, self.op, [c.data for c in self._prev]})"

a = Value(3)
b = Value(5)
c = a * b
d = b + a
e = c + d
print(e)
e.backprop()

print(a.grad)
print(b.grad)