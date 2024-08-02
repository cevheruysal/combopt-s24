import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from graph_algorithms import MaxFlowAlgorithms, MinSpanningTreeAlgorithms, UtilAlgorithms, MinDistanceAlgorithms
from enums import MinDistanceAlgorithmsEnum, EdgeDirection
from notation import Arc, Network, Vertex, Edge, Graph


class TestUtilAlgorithms(unittest.TestCase):

    def setUp(self):
        V = [Vertex(i) for i in range(1, 6)]

        e1 = Edge(1, 1, 2, 1, EdgeDirection.DIRECTED)
        e2 = Edge(2, 1, 3, 1, EdgeDirection.DIRECTED)
        e3 = Edge(3, 1, 4, 1, EdgeDirection.DIRECTED)
        e4 = Edge(4, 1, 5, 1, EdgeDirection.DIRECTED)
        e5 = Edge(5, 2, 4, 1, EdgeDirection.DIRECTED)
        e6 = Edge(6, 3, 4, 1, EdgeDirection.DIRECTED)
        e7 = Edge(7, 3, 5, 1, EdgeDirection.DIRECTED)
        e8 = Edge(8, 4, 5, 1, EdgeDirection.DIRECTED)

        self.g = Graph(1, V, E=[e1, e2, e3, e4, e5, e6, e7, e8])

    def test_topological_sort(self):
        order = UtilAlgorithms.topological_sort(self.g)
        self.assertEqual(order, [1, 2, 3, 4, 5])


class TestMinDistanceAlgorithms(unittest.TestCase):

    def setUp(self):
        self.v1 = Vertex(1)
        self.v2 = Vertex(2)
        self.v3 = Vertex(3)
        self.v4 = Vertex(4)
        self.e1 = Edge(1, 1, 2, 1, EdgeDirection.DIRECTED)
        self.e2 = Edge(2, 2, 3, 1, EdgeDirection.DIRECTED)
        self.e3 = Edge(3, 3, 4, 1, EdgeDirection.DIRECTED)
        self.e4 = Edge(4, 1, 3, 10, EdgeDirection.DIRECTED)
        self.g = Graph(1, [self.v1, self.v2, self.v3, self.v4], [self.e1, self.e2, self.e3, self.e4])
        self.algorithms = MinDistanceAlgorithms(self.g)

    def test_topological_sort_min_dist_algorithm(self):
        result = self.algorithms.topological_sort_min_dist_algorithm(1)
        expected = {1: 0, 2: 1, 3: 2, 4: 3}
        self.assertEqual(result, expected)

    def test_dijkstras_min_dist_algorithm(self):
        result = self.algorithms.dijkstras_min_dist_algorithm(1)
        expected = {1: 0, 2: 1, 3: 2, 4: 3}
        self.assertEqual(result, expected)

    def test_bellman_fords_min_dist_algorithm(self):
        result = self.algorithms.bellman_fords_min_dist_algorithm(1)
        expected = {1: 0, 2: 1, 3: 2, 4: 3}
        self.assertEqual(result, expected)

    def test_floyd_warshall_min_dist_algorithm(self):
        result = self.algorithms.floyd_warshall_min_dist_algorithm(1)
        expected = {1: 0, 2: 1, 3: 2, 4: 3}
        self.assertEqual(result, expected)

    def test_a_star_min_dist_algorithm(self):
        def heuristic(a, b):
            return 1  # Dummy heuristic for testing
        result = self.algorithms.a_star_min_dist_algorithm(1, 4, heuristic)
        expected = [1, 2, 3, 4]
        self.assertEqual(result, expected)

    @patch.object(MinDistanceAlgorithms, 'topological_sort_min_dist_algorithm')
    @patch.object(MinDistanceAlgorithms, 'dijkstras_min_dist_algorithm')
    @patch.object(MinDistanceAlgorithms, 'bellman_fords_min_dist_algorithm')
    def test_run(self, mock_bellman, mock_dijkstra, mock_topo):
        mock_topo.return_value = {1: 0, 2: 1, 3: 2, 4: 3}
        mock_dijkstra.return_value = {1: 0, 2: 1, 3: 2, 4: 3}
        mock_bellman.return_value = {1: 0, 2: 1, 3: 2, 4: 3}

        self.g.direction = True
        self.g.acyclical = True
        result = self.algorithms.run(1, 4, MinDistanceAlgorithmsEnum.TOPOLOGICAL_SORT)
        mock_topo.assert_called_once()
        self.assertEqual(result, {1: 0, 2: 1, 3: 2, 4: 3})

        self.g.direction = True
        self.g.has_negative_weight = False
        result = self.algorithms.run(1, 4, MinDistanceAlgorithmsEnum.DIJKSTRA)
        mock_dijkstra.assert_called_once()
        self.assertEqual(result, {1: 0, 2: 1, 3: 2, 4: 3})

        self.g.has_negative_weight = True
        result = self.algorithms.run(1, 4, MinDistanceAlgorithmsEnum.BELLMAN_FORD)
        mock_bellman.assert_called_once()
        self.assertEqual(result, {1: 0, 2: 1, 3: 2, 4: 3})


class TestMinSpanningTreeAlgorithms(unittest.TestCase):

    def setUp(self):
        undir = EdgeDirection.UNDIRECTED
        self.v1 = Vertex(1)
        self.v2 = Vertex(2)
        self.v3 = Vertex(3)
        self.v4 = Vertex(4)
        self.v5 = Vertex(5)
        self.e1 = Edge(1, 1, 2, 1, undir)
        self.e2 = Edge(2, 1, 3, 3, undir)
        self.e3 = Edge(3, 2, 3, 1, undir)
        self.e4 = Edge(4, 2, 4, 6, undir)
        self.e5 = Edge(5, 3, 4, 5, undir)
        self.e6 = Edge(6, 3, 5, 2, undir)
        self.e7 = Edge(7, 4, 5, 4, undir)
        self.g = Graph(1, [self.v1, self.v2, self.v3, self.v4, self.v5], [self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7])
        self.algorithms = MinSpanningTreeAlgorithms(self.g)

    def test_prims_min_spanning_tree_algorithm(self):
        mst = self.algorithms.prims_min_spanning_tree_algorithm()
        mst_edges = set(mst.edges.keys())
        expected_edges = {(1, 2), (2, 3), (3, 5), (4, 5)}
        self.assertEqual(mst_edges, expected_edges)

    def test_kruskals_min_spanning_tree_algorithm(self):
        mst = self.algorithms.kruskals_min_spanning_tree_algorithm()
        mst_edges = set(mst.edges)
        expected_edges = {(1, 2), (2, 3), (3, 5), (4, 5)}
        self.assertEqual(mst_edges, expected_edges)

    def test_kruskals_min_spanning_tree_algorithm_disconnected(self):
        v6 = Vertex(6)
        self.g.vertices[6] = v6
        mst = self.algorithms.kruskals_min_spanning_tree_algorithm()
        mst_edges = set(mst.edges)
        expected_edges = {(1, 2), (2, 3), (3, 5), (4, 5)}
        self.assertEqual(mst_edges, expected_edges)
        self.assertEqual(len(mst.edges), 4)

    def test_prims_min_spanning_tree_algorithm_single_vertex(self):
        single_vertex_graph = Graph(2, [Vertex(1)], [])
        single_vertex_algorithms = MinSpanningTreeAlgorithms(single_vertex_graph)
        mst = single_vertex_algorithms.prims_min_spanning_tree_algorithm()
        self.assertEqual(len(mst.edges), 0)

    def test_kruskals_min_spanning_tree_algorithm_single_vertex(self):
        single_vertex_graph = Graph(2, [Vertex(1)], [])
        single_vertex_algorithms = MinSpanningTreeAlgorithms(single_vertex_graph)
        mst = single_vertex_algorithms.kruskals_min_spanning_tree_algorithm()
        self.assertEqual(len(mst.edges), 0)

    def test_prims_min_spanning_tree_algorithm_no_edges(self):
        no_edge_graph = Graph(2, [Vertex(1), Vertex(2)], [])
        no_edge_algorithms = MinSpanningTreeAlgorithms(no_edge_graph)
        mst = no_edge_algorithms.prims_min_spanning_tree_algorithm()
        self.assertEqual(len(mst.edges), 0)

    def test_kruskals_min_spanning_tree_algorithm_no_edges(self):
        no_edge_graph = Graph(2, [Vertex(1), Vertex(2)], [])
        no_edge_algorithms = MinSpanningTreeAlgorithms(no_edge_graph)
        mst = no_edge_algorithms.kruskals_min_spanning_tree_algorithm()
        self.assertEqual(len(mst.edges), 0)


class TestMaxFlowAlgorithms(unittest.TestCase):

    def setUp(self):
        self.v1 = Vertex(1)
        self.v2 = Vertex(2)
        self.v3 = Vertex(3)
        self.v4 = Vertex(4)
        self.e1 = Arc(1, 1, 2, 3)
        self.e2 = Arc(2, 1, 3, 3)
        self.e3 = Arc(3, 2, 3, 2)
        self.e4 = Arc(4, 2, 4, 3)
        self.e5 = Arc(5, 3, 4, 4)
        self.n = Network(1, [self.v1, self.v2, self.v3, self.v4], 
                         [self.e1, self.e2, self.e3, self.e4, self.e5], 1, 4)
        self.algorithms = MaxFlowAlgorithms(self.n)

    def test_ford_fulkerson_max_flow_algorithm(self):
        max_flow = self.algorithms.ford_fulkerson_max_flow_algorithm()
        self.assertEqual(max_flow, 6)  # Expected max flow for this graph

    def test_edmonds_karp_max_flow_algorithm(self):
        max_flow = self.algorithms.edmonds_karp_max_flow_algorithm()
        self.assertEqual(max_flow, 6)  # Expected max flow for this graph

    def test_dinics_max_flow_algorithm(self):
        max_flow = self.algorithms.dinics_max_flow_algorithm()
        self.assertEqual(max_flow, 6)  # Expected max flow for this graph

    def test_ford_fulkerson_with_bottleneck(self):
        self.e1.capacity = 2  # Create a bottleneck
        max_flow = self.algorithms.ford_fulkerson_max_flow_algorithm()
        self.assertEqual(max_flow, 5)  # Expected max flow with bottleneck

    def test_edmonds_karp_with_disconnected_graph(self):
        self.e4.capacity = 0  # Disconnect part of the graph
        max_flow = self.algorithms.edmonds_karp_max_flow_algorithm()
        self.assertEqual(max_flow, 4)  # Expected max flow with disconnection

    def test_dinics_with_high_capacity_edges(self):
        self.e1.capacity = 10  # High capacity edge
        self.e2.capacity = 10
        max_flow = self.algorithms.dinics_max_flow_algorithm()
        self.assertEqual(max_flow, 7)  # Expected max flow with high capacity edge


if __name__ == '__main__':
    unittest.main()
