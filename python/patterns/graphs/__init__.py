from .core import DAG
try:
    from .graphtool import GraphTool
except ImportError:
    class GraphTool(DAG):

        def __init__(self, instance, resource):
            pass

        def available(self):
            return False
try:
    from .networkx import networkx
except ImportError:
    class networkx(DAG):

        def __init__(self, instance, resource):
            pass

        def available(self):
            return False

def graph_factory(instance, resource, options=None):
    if not options:
        options = {}
    g = GraphTool(instance, resource)
    if g.available():
        return g
    else:
        raise ImportError('There was a problem with GraphTool')

