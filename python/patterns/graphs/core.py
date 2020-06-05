import patterns.node as nd
import random as rn
import logging as log
import itertools
import numpy as np

class DAG(object):

    def __init__(self, instance, resource):
        self.source = nd.get_source_node(instance, resource)
        self.sink = nd.get_sink_node(instance, resource)
        self.instance = instance
        self.resource = resource
        # We generate the graph by using "nodes" module
        # We represent the graph with an adjacency list
        nodes_ady = self.source.walk_over_nodes()

        # Here, there is some post-processing (more nodes) to use the graph better
        # 1. for each period, for each assignment,
        # 2. tie all nodes to a single node with the same period, same assignment but rut and ret to None

        # We only create artificial nodes for all not None nodes.
        nodes_artificial = self.get_artificial_nodes(nodes_ady)
        self.g = nodes_ady.kvapply(lambda k, v: v + nodes_artificial.get(k, []))

        # this should deal with almost all nodes. maybe not the last one?
        self.refs = self.g.kapply(lambda k: k)
        self.refs_inv = self.refs

        pass

    def available(self):
        return True

    def draw(self):
        pass

    def shortest_path(self, node1=None, node2=None, **kwargs):
        pass

    def nodes_to_patterns(self, node1, node2, vp_not_task, cutoff=1000, max_paths=1000, add_empty=True,  **kwargs):
        pass

    def nodes_to_pattern(self, node1, node2, all_weights, cutoff=1000, **kwargs):
        pass

    @staticmethod
    def iter_sample_fast2(iterable, samplesize, max_iterations=9999999):
        """
        :param iter iterable:
        :param samplesize:
        :return:
        """
        universe = list(itertools.islice(iterable, max_iterations))
        if len(universe) > samplesize:
            return np.random.choice(universe, samplesize, replace=False).tolist()
        return universe

    @staticmethod
    def iter_sample_fast(iterable, samplesize, max_iterations=9999999):
        """

        :param iter iterable:
        :param samplesize:
        :return:
        # https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator
        """
        results = []
        iterator = iter(iterable)
        # Fill in the first samplesize elements:
        try:
            for _ in range(samplesize):
                results.append(next(iterator))
        except StopIteration:
            return results
        rn.shuffle(results)  # Randomize their positions
        # i = 0
        for i, v in enumerate(iterator, samplesize):
            # log.debug("iteration: {}".format(i))
            r = rn.randint(0, i)
            if r < samplesize:
                results[r] = v  # at a decreasing rate, replace random items
            if i >= max_iterations:
                break

        return results

    def get_all_patterns(self):
        return self.nodes_to_patterns(self.g, self.refs, self.refs_inv, self.source, self.sink)

    def get_artificial_nodes(self, nodes_ady):
        return \
            nodes_ady.keys_tl().\
            vfilter(lambda v: v.assignment is not None).\
            to_dict(None).\
            kvapply(lambda k, v: [nd.Node.from_node(node=k, rut=None, ret=None)])

    def to_file(self, path):
        pass

    def from_file(self, path, instance, resource):
        pass