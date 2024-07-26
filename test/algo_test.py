import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from graph_algorithms import UtilAlgorithms, MinDistanceAlgorithms
from enums import MinDistanceAlgorithmsEnum, EdgeDirection
from notation import Vertex, Edge, Graph

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


if __name__ == '__main__':
    unittest.main()
