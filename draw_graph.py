from graphviz import Digraph

def draw_graph(node):
    def add_nodes_edges(node, dot=None):
        if dot is None:
            dot = Digraph()
            dot.attr(rankdir='LR')  # Set the direction of the graph to left-to-right
            dot.node(str(id(node)), label=f"{{ {node.label} | {str(node.data).center(10)} | {str(node.grad).center(10)} }}", shape='record')
        
        if node.op:
            op_node_id = str(id(node)) + node.op
            dot.node(op_node_id, label=node.op)
            dot.edge(op_node_id, str(id(node)))
        
        for child in node.children:
            dot.node(str(id(child)), label=f"{{ {child.label} | data: {str(child.data).center(10)} | grad: {str(child.grad).center(10)} }}", shape='record')
            if node.op:
                dot.edge(str(id(child)), op_node_id)
            else:
                dot.edge(str(id(child)), str(id(node)))
            dot = add_nodes_edges(child, dot=dot)
        
        return dot

    dot = add_nodes_edges(node)
    return dot