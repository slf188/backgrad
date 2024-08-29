import math

class Scalar:
    def __init__(self, data, label=None, op="", children=None):
        self.data = data
        self.grad = 0.0
        self.op = op
        self.children = children if children is not None else []
        self.label = label
    
    def __repr__(self):
        return f"Scalar({self.data}, gradient {self.grad})"

    def __add__(self, other):
        if isinstance(other, Scalar):
            return Scalar(self.data + other.data, op="+", children=[self, other])
        else:
            return Scalar(self.data + other, op="+", children=[self])
        
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, Scalar):
            return Scalar(self.data * other.data, op="*", children=[self, other])
        else:
            return Scalar(self.data * other, op="*", children=[self])
        
    def __rmul__(self, other):
        return self * other
    
    def __sub__(self, other):
        if isinstance(other, Scalar):
            return Scalar(self.data - other.data, op="-", children=[self, other])
        else:
            return Scalar(self.data - other, op="-", children=[self])
        
    def __rsub__(self, other):
        return Scalar(other) - self
    
    def __truediv__(self, other):
        if isinstance(other, Scalar):
            return Scalar(self.data / other.data, op="/", children=[self, other])
        else:
            return Scalar(self.data / other, op="/", children=[self])
        
    def __rtruediv__(self, other):
        return Scalar(other) / self
    
    def __pow__(self, other):
        if isinstance(other, Scalar):
            return Scalar(self.data ** other.data, op="**", children=[self, other])
        else:
            return Scalar(self.data ** other, op="**", children=[self])
        
    def __rpow__(self, other):
        return Scalar(other) ** self
    
    def backward(self):
        # Topological ordering of all children in the graph
        topo_order = []
        visited = set()
        
        def build_topo_order(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topo_order(child)
                topo_order.append(node)
        
        build_topo_order(self)

        # Initialize the gradient of the final output scalar
        self.grad = 1.0
        
        # Backpropagate through the graph
        for node in reversed(topo_order):
            # Depending on the operation, calculate gradients
            # sum
            if node.op == "+":
                for child in node.children:
                    child.grad += 1.0 * node.grad  # gradient flows directly through addition
            # substraction
            elif node.op == "-":
                node.children[0].grad += 1.0 * node.grad  # For left operand
                node.children[1].grad += -1.0 * node.grad  # For right operand
            # multiplication
            elif node.op == "*":
                node.children[0].grad += node.children[1].data * node.grad  # grad of first operand
                node.children[1].grad += node.children[0].data * node.grad  # grad of second operand
            # division
            elif node.op == "/":
                node.children[0].grad += (1 / node.children[1].data) * node.grad  # grad of numerator
                node.children[1].grad += -(node.children[0].data / node.children[1].data ** 2) * node.grad  # grad of denominator
            # power
            elif node.op == "**":
                base, exponent = node.children
                base.grad += (exponent.data * (base.data ** (exponent.data - 1))) * node.grad  # grad w.r.t base
                exponent.grad += (node.data * math.log(base.data)) * node.grad  # grad w.r.t exponent

def print_graph(node, indent=0):
    print("  " * indent + f"{node.label} | data {node.data:.4f} | grad {node.grad:.4f} | op {node.op}")
    for child in node.children:
        print_graph(child, indent=indent+1)

# restart the gradients
def restart_graph(node):
    node.grad = 0.0
    for child in node.children:
        restart_graph(child)
