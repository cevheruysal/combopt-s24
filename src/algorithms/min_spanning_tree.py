import heapq
import logging
from random import choice

from src.enums import MinSpanningTreeAlgorithmsEnum
from src.graph_utils import delta, minimum_cost_edge_in_delta
from src.notation import Graph, Tree
from src.util_structs import UnionFind

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MinSpanningTreeAlgorithms:
    def __init__(self, G: Graph):
        """
        Let (𝐺, 𝑐) be an instance of the minimum spanning tree problem with 𝐺 = (𝑉, 𝐸),
        and let 𝑇 = (𝑉, 𝐸_𝑇) be a spanning tree in 𝐺.
        Then the following statements are equivalent:

        (i) 𝑇 is a minimum spanning tree with respect to 𝑐.
        (ii) For every edge 𝑒 = {𝑢, 𝑣} ∈ 𝐸 ∖ 𝐸_𝑇, no edge on the 𝑢-𝑣-path in 𝑇 has a higher cost than 𝑐(𝑒).
        (iii) For every 𝑒 ∈ 𝐸_𝑇, and every connected component 𝐶 of 𝑇 − 𝑒 the cost 𝑐(𝑒) is minimum in 𝛿_𝐺(𝑉(𝐶)).
        (iv) We can order 𝐸_𝑇 = {𝑒_1, … , 𝑒_𝑛−1} such that for each 𝑖 ∈ {1, … , 𝑛 − 1} there exists a subset 𝑉′ ⊂ 𝑉
        of the nodes such that 𝑒_𝑖 is a minimum cost edge of 𝛿_𝐺(𝑉′) and 𝑒_𝑗 ∉ 𝛿_𝐺(𝑉′) for all 𝑗 ∈ {1, … , 𝑖 − 1} """

        self.graph = G

    def run(self, use_algorithm: MinSpanningTreeAlgorithmsEnum = MinSpanningTreeAlgorithmsEnum.KRUSKALS) -> Graph:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find minimum spanning tree")

            match use_algorithm.value:
                case 1:
                    result = self.prims_min_spanning_tree_algorithm()
                case 2:
                    result = self.kruskals_min_spanning_tree_algorithm()
                case _:
                    logger.error("Algorithm doesn't exist returning no solution")

        return result

    def prims_min_spanning_tree_algorithm(self):
        """
        input : a graph 𝐺 = (𝑉, 𝐸), edge weights 𝑐 ∶ 𝐸 → Q≥0
        output: a minimum spanning tree 𝑇 = (𝑉, 𝐸_𝑇) in 𝐺

        1 choose an arbitrary node 𝑣0 ∈ 𝑉
        2 initialize 𝑇 = (𝑉_𝑇, 𝐸_𝑇) ∶= ({𝑣_0}, ∅)
        3 while 𝑉_𝑇 ≠ 𝑉 do
            4 choose an edge 𝑒 ∈ 𝛿_𝐺(𝑉_𝑇) of minimum weight 𝑐(𝑒)
            5 set 𝑇 := 𝑇 + 𝑒
        return T

        Prim’s algorithm works correctly, i. e., it outputs a minimal spanning tree.
        It can be implemented to run in time 𝒪(𝑛^2) """

        v_0 = choice(self.graph.vertices)
        T = Tree(Id="MST_Prims", V=[v_0], E=[])

        while len(T.vertices) < len(self.graph.vertices):
            edge_list, from_vertices, to_vertices = (self.graph.edges, T.vertices.keys(),
                                                     [v for v in self.graph.vertices.keys() if
                                                      v not in T.vertices.keys()])
            delta_edges = delta(edge_list, from_vertices, to_vertices)
            min_cost_edge = minimum_cost_edge_in_delta(delta_edges)
            if min_cost_edge is None: break
            T.add_edge(min_cost_edge)

        if len(T.vertices) < len(self.graph.vertices):
            logger.warning("Tree doesn't span the entirety of the Graph!!")

        return T

    def kruskals_min_spanning_tree_algorithm(self):
        """
        input : a graph 𝐺 = (𝑉, 𝐸), edge weights 𝑐 ∶ 𝐸 → Q≥0
        output: a minimum spanning tree 𝑇 = (𝑉, 𝐸_𝑇) in 𝐺

        1 sort the edges 𝐸 = {𝑒_1, … , 𝑒_𝑚} of 𝐺 such that 𝑐(𝑒_1) ≤ 𝑐(𝑒_2) ≤ ⋯ ≤ 𝑐(𝑒_𝑚).
        2 initialize 𝑇 = (𝑉_𝑇, 𝐸_𝑇) ∶= (𝑉, ∅)
        3 for 𝑖 = 1, … , 𝑚 do
            4 if 𝑇 + 𝑒_𝑖 does not contain a cycle then
            5 set 𝑇 ∶= 𝑇 + 𝑒_𝑖
        6 return 𝑇

        Kruskal’s algorithm works correctly and can be implemented in time 𝒪(𝑚*𝑛) """

        T = Tree(Id="MST_Kruskals", V=[], E=[])
        uf = UnionFind(self.graph.vertices.keys())
        edge_heap = [(e.weight, e.copy()) for e in self.graph.edges.values()]
        heapq.heapify(edge_heap)

        while len(edge_heap) > 0:
            _, e = heapq.heappop(edge_heap)
            v1, v2 = e.incident_vertex_ids

            if uf.find(v1) != uf.find(v2):
                T.add_edge(e)

        if len(T.vertices) < len(self.graph.vertices):
            logger.warning("Tree doesn't span the entirety of the Graph!!")

        return T
