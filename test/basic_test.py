import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from notation import Vertex, Edge, Graph, Forest, Tree  # Adjust the import path as necessary
from enums import EdgeDirection, GraphDirection


class TestVertex(unittest.TestCase):
    
    def test_vertex_creation(self):
        v = Vertex(1)
        self.assertEqual(v.id, 1)
        self.assertEqual(v.roots, set())
        self.assertEqual(v.leafs, set())

    def test_vertex_copy(self):
        v = Vertex(1, {2, 3}, {4, 5})
        v_copy = v.copy()
        self.assertEqual(v.id, v_copy.id)
        self.assertEqual(v.roots, v_copy.roots)
        self.assertEqual(v.leafs, v_copy.leafs)
        self.assertEqual(v, v_copy)

    def test_vertex_equality(self):
        v1 = Vertex(1)
        v2 = Vertex(1)
        v3 = Vertex(2)
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)

    def test_vertex_str(self):
        v = Vertex(0, {1, 2, 3}, {4, 5})
        expected_output = "Roots: v1, v2, v3\n" +\
                          "Id:       V0\n" +\
                          "Leafs: v4, v5\n"
        self.assertEqual(str(v), expected_output)


class TestEdge(unittest.TestCase):
    
    def test_edge_creation(self):
        e = Edge(1, 2, 3, 4.5)
        self.assertEqual(e.id, 1)
        self.assertEqual(e.end_vertex_ids, (2, 3))
        self.assertEqual(e.weight, 4.5)
        self.assertEqual(e.direction, EdgeDirection.DIRECTED)

    def test_edge_copy(self):
        e1 = Edge(1, 2, 3, 4.5)
        e2 = Edge(2, 2, 3, 1)
        e1_copy = e1.copy()
        self.assertEqual(e1.id, e1_copy.id)
        self.assertEqual(e1.end_vertex_ids, e1_copy.end_vertex_ids)
        self.assertEqual(e1.weight, e1_copy.weight)
        self.assertEqual(e1.direction, e1_copy.direction)
        self.assertEqual(e1, e1_copy)
        self.assertNotEqual(e1, e2)

    def test_edge_str(self):
        e1 = Edge(1, 2, 3, 4.5, EdgeDirection.DIRECTED)
        expected_output = "E1:= V2 ----W:+4.5---> V3\n"
        self.assertEqual(str(e1), expected_output)

        e2 = Edge(2, 4, 2, -4.5, EdgeDirection.BIDIRECTED)
        expected_output_2 = "E2:= V4 <---W:-4.5---> V2\n"
        self.assertEqual(str(e2), expected_output_2)

        e3 = Edge(3, 1, 5, -1, EdgeDirection.UNDIRECTED)
        expected_output_3 = "E3:= V1 ----W:-1.0---- V5\n"
        self.assertEqual(str(e3), expected_output_3)


class TestGraph(unittest.TestCase):

    def test_graph_creation(self):
        v1 = Vertex(1)
        v2 = Vertex(2)
        e1 = Edge(1, 1, 2, 4.5)
        e2 = Edge(2, 2, 1, -1)
        g = Graph(1, [v1, v2], [e1, e2])
        self.assertEqual(g.id, 1)
        self.assertEqual(len(g.vertices), 2)
        self.assertEqual(len(g.edges), 2)

    def test_graph_init_edge(self):
        v1 = Vertex(1)
        v2 = Vertex(2)
        e = Edge(1, 1, 2, 4.5)
        g = Graph(1, [v1, v2], [])
        g.init_edge(e)
        self.assertEqual(len(g.edges), 1)
        self.assertIn((1, 2), g.edges)
        self.assertIn(2, g.vertices[1].leafs)
        self.assertIn(1, g.vertices[2].roots)

    def test_graph_init_edge_2(self):
        v1, v2, v3 = Vertex(1), Vertex(2), Vertex(3)

        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 3, -1, EdgeDirection.BIDIRECTED)
        e3 = Edge(3, 1, 3, 3, EdgeDirection.UNDIRECTED)
        
        g = Graph(1, [v1, v2], [e1, e2])
        g.init_edge(e3)
        self.assertEqual(len(g.edges), 3)
        self.assertIn((1, 3), g.edges)
        self.assertIn(3, g.vertices[1].roots)
        self.assertIn(2, g.vertices[1].leafs)
        self.assertIn(3, g.vertices[1].leafs)
        
        self.assertIn(1, g.vertices[2].roots)
        self.assertIn(3, g.vertices[2].roots)
        self.assertIn(3, g.vertices[2].leafs)
        
        self.assertIn(1, g.vertices[3].roots)
        self.assertIn(2, g.vertices[3].roots)
        self.assertIn(1, g.vertices[3].leafs)
        self.assertIn(2, g.vertices[3].leafs)

    def test_graph_direction(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 3, 1, EdgeDirection.DIRECTED)
        e3 = Edge(3, 3, 4, -1, EdgeDirection.BIDIRECTED)
        e4 = Edge(3, 3, 4, -1, EdgeDirection.UNDIRECTED)

        g = Graph(1, [], [e1])
        self.assertEqual(g.get_graph_direction(), GraphDirection.DIRECTED)
        g.init_edge(e2)
        self.assertEqual(g.get_graph_direction(), GraphDirection.DIRECTED)
        g.init_edge(e3)
        self.assertEqual(g.get_graph_direction(), GraphDirection.MIXED)

        g2 = Graph(2, [], [e4])
        self.assertEqual(g2.get_graph_direction(), GraphDirection.UNDIRECTED)


    def test_digraph_isCyclic(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 1, 4.5, EdgeDirection.DIRECTED)
        g = Graph(1, E=[e1, e2])
        self.assertTrue(g.isCyclic())


    def test_ungraph_isNotCyclic(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.UNDIRECTED)
        e2 = Edge(2, 2, 1, 4.5, EdgeDirection.UNDIRECTED)
        g = Graph(1, E=[e1, e2])
        self.assertFalse(g.isCyclic())

    
    def test_bigraph_isNotCyclic(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.BIDIRECTED)
        e2 = Edge(2, 2, 1, 4.5, EdgeDirection.BIDIRECTED)
        g = Graph(1, E=[e1, e2])
        self.assertFalse(g.isCyclic())


    def test_digraph_isCyclic2(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 3, 4.5, EdgeDirection.DIRECTED)
        e3 = Edge(3, 3, 1, 4.5, EdgeDirection.DIRECTED)
        
        g = Graph(1, E=[e1, e2, e3])
        self.assertTrue(g.isCyclic())
        

    def test_digraph_isCyclic3(self):
        V = [Vertex(i) for i in range(1, 9)]

        e1 = Edge(1, 1, 2, 1, EdgeDirection.DIRECTED)
        e2 = Edge(2, 1, 3, 1, EdgeDirection.DIRECTED)
        e3 = Edge(3, 1, 4, 1, EdgeDirection.DIRECTED)
        e4 = Edge(4, 1, 5, 1, EdgeDirection.DIRECTED)
        e5 = Edge(5, 2, 4, 1, EdgeDirection.DIRECTED)
        e6 = Edge(6, 3, 4, 1, EdgeDirection.DIRECTED)
        e7 = Edge(7, 3, 5, 1, EdgeDirection.DIRECTED)
        e8 = Edge(8, 4, 5, 1, EdgeDirection.DIRECTED)

        g = Graph(1, V, E=[e1, e2, e3, e4, e5, e6, e7, e8])
        self.assertFalse(g.isCyclic())


    def test_graph_str(self):
        e = Edge(1, 1, 2, 4.5)
        g = Graph(1, E=[e])
        expected_output = "Metadata of Graph1:\nThis is an directed acyclic graph with all positive edge weights\n"
        self.assertEqual(str(g), expected_output)


class TestForest(unittest.TestCase):

    def test_forest_creation(self):
        v1 = Vertex(1)
        v2 = Vertex(2)
        e = Edge(1, 1, 2, 4.5)
        f = Forest(1, [v1, v2], [e])
        self.assertEqual(f.id, 1)
        self.assertEqual(len(f.vertices), 2)
        self.assertEqual(len(f.edges), 1)

    def test_forest_isCyclic(self):
        v1 = Vertex(1)
        v2 = Vertex(2)
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 1, 4.5, EdgeDirection.DIRECTED)
        with self.assertRaises(ValueError):
            Forest(1, [v1, v2], [e1, e2])


class TestTree(unittest.TestCase):

    def test_tree_creation(self):
        v1 = Vertex(1)
        v2 = Vertex(2)
        e = Edge(1, 1, 2, 4.5)
        t = Tree(1, [v1, v2], [e])
        self.assertEqual(t.id, 1)
        self.assertEqual(len(t.vertices), 2)
        self.assertEqual(len(t.edges), 1)

    def test_tree_is_connected(self):
        v1 = Vertex(1)
        v2 = Vertex(2)
        v3 = Vertex(3)
        e = Edge(1, 1, 2, 4.5)
        with self.assertRaises(ValueError):
            Tree(1, [v1, v2, v3], [e])


if __name__ == '__main__':
    unittest.main()
