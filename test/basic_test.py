import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from enums import EdgeDirection, GraphDirection
from notation import Arc, Edge, Forest, Graph, Network, Tree, Vertex


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
        expected_output = "Roots: v1, v2, v3\n" + "Id:       V0\n" + "Leafs: v4, v5\n"
        self.assertEqual(str(v), expected_output)


class TestEdge(unittest.TestCase):

    def test_edge_creation(self):
        e = Edge(1, 2, 3, 4.5)
        self.assertEqual(e.id, 1)
        self.assertEqual(e.incident_vertex_ids, (2, 3))
        self.assertEqual(e.weight, 4.5)
        self.assertEqual(e.direction, EdgeDirection.DIRECTED)

    def test_edge_copy(self):
        e1 = Edge(1, 2, 3, 4.5)
        e2 = Edge(2, 2, 3, 1)
        e1_copy = e1.copy()
        self.assertEqual(e1.id, e1_copy.id)
        self.assertEqual(e1.incident_vertex_ids, e1_copy.incident_vertex_ids)
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


class TestArc(unittest.TestCase):

    def test_arc_initialization(self):
        a = Arc(1, 1, 2, 10.0, 5.0)
        self.assertEqual(a.id, 1)
        self.assertEqual(a.incident_vertex_ids, (1, 2))
        self.assertEqual(a.capacity, 10.0)
        self.assertEqual(a.flow, 5.0)
        self.assertEqual(a.residual_arc, False)

    def test_arc_remaining_capacity(self):
        a = Arc(1, 1, 2, 10.0, 5.0)
        self.assertEqual(a.remaining_capacity(), 5.0)

        a_res = Arc(1, 2, 1, 4.0, 0.0, True)
        self.assertEqual(a_res.remaining_capacity(), 4.0)

    def test_arc_flow_methods(self):
        a_id, a_res_id, v1, v2, a_cap, a_flow = 0, 1, 1, 2, 10.0, 5.0
        a = Arc(a_id, v1, v2, a_cap, a_flow)
        a_res = Arc(a_res_id, v2, v1, a_flow, 0.0, True)

        f = 4.5
        dF = -2.3  # any float less than a.remaining_capacity()

        a.set_flow(f)
        a_res.set_flow(f)

        self.assertEqual(a.flow, f)
        self.assertEqual(a.capacity, a_cap)

        self.assertEqual(a_res.flow, 0.0)
        self.assertEqual(a_res.capacity, f)

        a.alter_flow(dF)
        a_res.alter_flow(-dF)

        self.assertEqual(a.flow, a_res.capacity)

    def test_arc_copy(self):
        a1 = Arc(1, 1, 2, 10.0, 5.0)
        a2 = a1.copy()
        self.assertEqual(a1, a2)
        a2.flow = 7.0
        self.assertNotEqual(a1, a2)

    def test_arc_string_representation(self):
        a = Arc(1, 1, 2, 10.0, 5.0)
        a_res1 = Arc(2, 2, 1, 5.0, 0.0, True)
        a_res2 = Arc(3, 2, 1, 0.0, 0.0, True)

        expected_output_0 = "A1:= V1 ----F/U:5.0/10.0---> V2\n"
        expected_output_1 = "A2:= V2 ----F/U:0.0/5.0---> V1\n"

        self.assertEqual(str(a), expected_output_0)
        self.assertEqual(str(a_res1), expected_output_1)
        self.assertEqual(str(a_res2), "")


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

    def test_graph_add_edge_and_vertex_leaf_and_root(self):
        v1, v2, v3 = Vertex(1), Vertex(2), Vertex(3)

        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 3, -1, EdgeDirection.BIDIRECTED)
        e3 = Edge(3, 1, 3, 3, EdgeDirection.UNDIRECTED)

        g = Graph(1, [v1, v2], [e1, e2])
        g.add_edge(e3)

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

    def test_graph_connected_components(self):
        v1, v2, v3 = Vertex(1), Vertex(2), Vertex(3)
        e1 = Edge(1, 1, 2, 10.0)
        g = Graph(1, [v1, v2, v3], [e1])
        self.assertEqual(g.get_connected_components(), 2)

    def test_graph_direction(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 3, 1, EdgeDirection.DIRECTED)
        e3 = Edge(3, 3, 4, -1, EdgeDirection.BIDIRECTED)
        e4 = Edge(3, 3, 4, -1, EdgeDirection.UNDIRECTED)

        g = Graph(1, [], [e1])
        self.assertEqual(g.get_graph_direction(), GraphDirection.DIRECTED)
        g.add_edge(e2)
        self.assertEqual(g.get_graph_direction(), GraphDirection.DIRECTED)
        g.add_edge(e3)
        self.assertEqual(g.get_graph_direction(), GraphDirection.MIXED)

        g2 = Graph(2, [], [e4])
        self.assertEqual(g2.get_graph_direction(), GraphDirection.UNDIRECTED)

    def test_digraph_isCyclic(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 1, 4.5, EdgeDirection.DIRECTED)
        g = Graph(1, E=[e1, e2])
        self.assertTrue(g.is_cyclic())

    def test_ungraph_isNotCyclic(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.UNDIRECTED)
        e2 = Edge(2, 2, 1, 4.5, EdgeDirection.UNDIRECTED)
        g = Graph(1, E=[e1, e2])
        self.assertFalse(g.is_cyclic())

    def test_bigraph_isNotCyclic(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.BIDIRECTED)
        e2 = Edge(2, 2, 1, 4.5, EdgeDirection.BIDIRECTED)
        g = Graph(1, E=[e1, e2])
        self.assertFalse(g.is_cyclic())

    def test_digraph_isCyclic2(self):
        e1 = Edge(1, 1, 2, 4.5, EdgeDirection.DIRECTED)
        e2 = Edge(2, 2, 3, 4.5, EdgeDirection.DIRECTED)
        e3 = Edge(3, 3, 1, 4.5, EdgeDirection.DIRECTED)

        g = Graph(1, E=[e1, e2, e3])
        self.assertTrue(g.is_cyclic())

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
        isCyclic = g.is_cyclic()
        self.assertFalse(isCyclic)
        self.assertEqual(isCyclic, not g.acyclical)

    def test_graph_copy(self):
        v1, v2 = Vertex(1), Vertex(2)
        e = Edge(1, 1, 2, 10.0)
        g1 = Graph(1, [v1, v2], [e])
        g2 = g1.copy()

        self.assertEqual(g1.id, g2.id)
        self.assertEqual(g1.edges[(1, 2)], g2.edges[(1, 2)])
        g2.edges[(1, 2)].weight = 20.0
        self.assertNotEqual(g1.edges[(1, 2)], g2.edges[(1, 2)])

    def test_graph_str(self):
        e = Edge(1, 1, 2, 4.5)
        g = Graph(1, E=[e])
        expected_output = "Metadata of Graph1:\nThis is a directed acyclic graph with all positive edge weights\n"
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


class TestNetwork(unittest.TestCase):

    def test_network_initialization(self):
        v1, v2 = Vertex(1), Vertex(2)
        a = Arc(1, 1, 2, 10.0, 5.0)
        n = Network(1, [v1, v2], [a], 1, 2)
        self.assertEqual(n.id, 1)
        self.assertTrue(n.st_connected)

    def test_network_residual_arcs(self):
        v1, v2, v3 = Vertex(1), Vertex(2), Vertex(3)
        a1 = Arc(1, 1, 2, 10.0, 5.0)
        a2 = Arc(2, 2, 3, 7.0)
        n = Network(1, [v1, v2, v3], [a1, a2], 1, 3)
        self.assertEqual(len(n.edges), 4)  # Includes 2 original and 2 residual arcs

    def test_network_flow_augmentation(self):
        v1, v2, v3 = Vertex(1), Vertex(2), Vertex(3)
        a1 = Arc(1, 1, 2, 10.0, 5.0)
        a2 = Arc(2, 2, 3, 7.0)
        n = Network(1, [v1, v2, v3], [a1, a2], 1, 3)
        augmenting_path = [(1, 2), (2, 3)]
        n.augment_along(augmenting_path, 5)
        self.assertEqual(n.flow, 5)
        self.assertEqual(n.edges[(1, 2)].flow, 10.0)


if __name__ == "__main__":
    unittest.main()
