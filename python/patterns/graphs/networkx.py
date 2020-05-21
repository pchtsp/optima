import networkx as nx
from pytups import superdict as sd, tuplist as tl
import patterns.node as nd
import random as rn
import numpy as np
import data.data_input as di
import os


class networkx(object):

    def get_create_node(self, node):
        try:
            return self.refs[node]
        except KeyError:
            v = self.g.add_vertex()
            self.refs[node] = int(v)
            return v

    def __init__(self, instance, resource):
        super().__init__(instance, resource)
        nodes_ady = self.g
        self.g = nx.Graph()

        # add nodes
        self.g.add_nodes_from(nodes_ady.keys())
        self.g.add_nodes_from(nodes_ady.keys())

        self.refs = sd.SuperDict()
        for n, n2_list in nodes_ady.items():
            v1 = self.get_create_node(n)
            for n2 in n2_list:
                v2 = self.get_create_node(n2)
                e = self.g.add_edge(v1, v2, add_missing=False)
        self.refs_inv = self.refs.reverse()

        # delete nodes with infinite distance to sink:
        self.g.set_reversed(is_reversed=True)
        distances = self.shortest_path(node1=self.sink)
        max_dist = instance.get_dist_periods(*instance.get_first_last_period()) + 5
        nodes = [n for n in self.g.vertices()
                 if distances[n] > max_dist and
                 self.refs_inv[n].rut is not None]
        self.g.set_reversed(is_reversed=False)
        self.g.remove_vertex(nodes, fast=True)
        # TODO: there is an issue with storing the GraphView and not the original object
        #  for now, we delete the nodes...
        # keep_node = self.g.new_vp('bool', val=1)
        # for v in nodes:
        #     keep_node[v] = 0
        # # self._g = self.g
        # self.g = gr.GraphView(self.g, vfilt=keep_node)

        # create dictionary to filter nodes that have no tasks
        self.vp_not_task = self.g.new_vp('bool', val=1)
        for v in self.g.vertices():
            if self.refs_inv[v].type == nd.TASK_TYPE:
                self.vp_not_task[v] = 0

    def draw(self):
        pass

    def shortest_path(self, node1=None, node2=None, **kwargs):
        pass

    def nodes_to_patterns(self, node1, node2, vp_not_task, cutoff=1000, max_paths=1000, add_empty=True,  **kwargs):
        pass

    def nodes_to_pattern(self, node1, node2, all_weights, cutoff=1000, **kwargs):
        pass

    def get_artificial_nodes(self, nodes_ady):
        return \
            nodes_ady.keys_tl().\
            vfilter(lambda v: v.assignment is not None).\
            vapply(lambda v: (v.period, v.period_end, v.assignment, v.type, v)).\
            to_dict(result_col=4).list_reverse().vapply(lambda v: v[0]).\
            vapply(lambda v: dict(instance=self.instance, resource=self.resource,
                                  period=v[0], period_end=v[1], assignment=v[2],
                                  rut=None, ret=None, type=v[3])).\
                vapply(lambda v: [nd.Node(**v)])

    def to_file(self, path):
        pass

    def from_file(self, path, instance, resource):
        pass
