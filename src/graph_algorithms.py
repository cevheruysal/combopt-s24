import heapq
import logging
from collections import deque
from random import choice
from typing import Dict, List, Optional, Tuple

import numpy as np

from enums import (MaxFlowAlgorithmsEnum, MinDistanceAlgorithmsEnum,
                   MinSpanningTreeAlgorithmsEnum)
from graph_utils import (construct_path_to_node, delta,
                         minimum_cost_edge_in_delta)
from notation import Graph, Network, Tree, Vertex
from util_structs import UnionFind

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class UtilAlgorithms:
    @staticmethod
    def topological_sort(G: Graph) -> Optional[List[int]]:
        if len(G.edges) == 0:
            logger.warning("No edges present to perform topological sorting")
            return None

        topological_order = []
        in_degree = {v.id: len(v.roots) for v in G.vertices.values()}
        queue = deque([v_id for v_id, deg in in_degree.items() if deg == 0])

        while queue:
            root = queue.popleft()
            topological_order.append(root)

            for leaf in G.vertices[root].leafs:
                in_degree[leaf] -= 1
                if in_degree[leaf] == 0:
                    queue.append(leaf)

        if len(topological_order) == len(G.vertices):
            return topological_order

        logger.info(
            "Topological sorting finished, no possible sorting found. The graph may have cycles."
        )
        return None

    @staticmethod
    def find_st_path(N: Network) -> Optional[Tuple[List[Tuple[int, int]], float]]:
        parent_dict = {v: -1 for v in N.vertices.keys()}
        visited = set()
        queue = deque([(N.source_node_id, float("inf"))])

        while queue:
            current, flow = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for neighbor in N.vertices[current].leafs:
                arc = N.edges[(current, neighbor)]

                if neighbor not in visited and arc.remaining_capacity() > 0:
                    new_flow = min(flow, arc.remaining_capacity())
                    parent_dict[neighbor] = current

                    if neighbor == N.sink_node_id:
                        path = construct_path_to_node(parent_dict, N.sink_node_id)
                        logger.info(f"Found augmenting path:{path} with flow {new_flow}")
                        return path, new_flow

                    queue.appendleft((neighbor, new_flow))
        return None, 0

    @staticmethod
    def blocking_flow(N: Network, node: int, flow: int, start: Dict[int, int]) -> int:
        if node == N.sink_node_id:
            return flow

        for leaf in N.vertices[node].leafs:
            # while start[node] < len(N.vertices[node].leafs):
            #     leaf = N.vertices[node].leafs[start[node]]

            if (
                N.node_levels[leaf] == N.node_levels[node] + 1
                and N.edges[(node, leaf)].remaining_capacity() > 0
            ):
                curr_flow = min(flow, N.edges[(node, leaf)].remaining_capacity())
                temp_flow = UtilAlgorithms.blocking_flow(N, leaf, curr_flow, start)

                if temp_flow > 0:
                    N.edges[(node, leaf)].alter_flow(temp_flow)
                    N.edges[(leaf, node)].alter_flow(-temp_flow)
                    return temp_flow

            start[node] += 1
        return 0


class MinDistanceAlgorithms:
    def __init__(self, G: Graph):
        self.graph = G

    def auto_run(self, start_vertex: int, goal_vertex: int) -> Optional[dict]:
        result = None
        if self.graph.direction and self.graph.acyclical:
            logger.info("The graph is directed and acyclic, using Topological Sort to find minimum distance")
            result = self.topological_sort_min_dist_algorithm(start_vertex)
        elif self.graph.direction and not self.graph.has_negative_weight:
            logger.info("The graph is directed and doesn't have negative weights, using Dijkstra's to find minimum distance")
            result = self.dijkstras_min_dist_algorithm(start_vertex)
        elif self.graph.has_negative_weight:
            logger.info("The graph is either undirected or there is negative weighted edges, using Bellman-Ford's to find minimum distance")
            result = self.bellman_fords_min_dist_algorithm(start_vertex)
        else:
            logger.error("Automatic selection didn't run any minimum distance algorithm")
        if result is None: 
            logger.error("Automatic selection couldn't find any solutions") 
        return result

    def run(self, 
            start_vertex:int, goal_vertex:int, 
            use_algorithm:MinDistanceAlgorithmsEnum = MinDistanceAlgorithmsEnum.AUTOMATIC) -> Optional[dict]:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find minimum distance")

            match use_algorithm.value:
                case 1: result = self.topological_sort_min_dist_algorithm(start_vertex)
                case 2: result = self.dijkstras_min_dist_algorithm(start_vertex)
                case 3: result = self.bellman_fords_min_dist_algorithm(start_vertex)
                case 4: result = self.floyd_warshall_min_dist_algorithm(start_vertex)
                case 5: result = self.a_star_min_dist_algorithm(start_vertex, goal_vertex)
                case _: logger.error("Algorithm doesn't exist returning no solution")
        else: 
            result = self.auto_run(start_vertex, goal_vertex)

        return result

    def topological_sort_min_dist_algorithm(self, 
                                            start_vertex: int) -> dict:
        """
        Let 𝐺 be an acyclic directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q and let 𝑠, 𝑡 ∈ 𝑉(𝐺).
        Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑛 + 𝑚)"""

        distances = {v.id: float("inf") for v in self.graph.vertices.values()}
        distances[start_vertex] = 0

        for vertex_id in UtilAlgorithms.topological_sort(self.graph):
            if distances[vertex_id] == float("inf"):
                continue

            for neighbor in self.graph.vertices[vertex_id].leafs:
                edge = self.graph.edges[(vertex_id, neighbor)]
                new_distance = distances[vertex_id] + edge.weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance

        logger.info("Found solution using Topological Sort")
        return distances

    def dijkstras_min_dist_algorithm(self, 
                                     start_vertex: int) -> dict:
        """
        Let 𝐺 be a directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q≥0 and let 𝑠, 𝑡 ∈ 𝑉(𝐺).
        Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑚 + 𝑛 log 𝑛)"""

        distances = {v.id: float("inf") for v in self.graph.vertices.values()}
        distances[start_vertex] = 0
        heap = [(0, start_vertex)]
        heapq.heapify(heap)

        while heap:
            current_distance, current_vertex = heapq.heappop(heap)
            if current_distance > distances[current_vertex]:
                continue

            for neighbor in self.graph.vertices[current_vertex].leafs:
                edge = self.graph.edges[(current_vertex, neighbor)]
                distance = current_distance + edge.weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))

        logger.info("Found solution using Dijkstra's")
        return distances

    def bellman_fords_min_dist_algorithm(self, 
                                         start_vertex: int) -> Optional[dict]:
        """
        Let 𝐺 = (𝑉, 𝐸) be a directed graph with edge weights 𝑐 ∶ 𝐸 → Q and let 𝑠, 𝑡 ∈ 𝑉.
        There is an algorithm that either computes a shortest 𝑠-𝑡-path in 𝐺
        or finds a negative cycle in 𝐺 in time 𝒪 (𝑚𝑛)"""

        distances = {v.id: float("inf") for v in self.graph.vertices.values()}
        distances[start_vertex] = 0

        for _ in range(len(self.graph.vertices) - 1):
            for edge in self.graph.edges.values():
                u, v = edge.incident_vertex_ids
                distances[v] = min(distances[u] + edge.weight, distances[v])

        # Check for negative weight cycles
        for edge in self.graph.edges.values():
            u, v = edge.incident_vertex_ids
            if distances[u] + edge.weight < distances[v]:
                logger.info("Graph contains a negative weight cycle")
                return None

        logger.info("Found solution using Bellman Ford's")
        return distances

    def floyd_warshall_min_dist_algorithm(self, 
                                          start_vertex: int) -> Optional[dict]:
        v_size = len(self.graph.vertices)
        distance_matrix = np.ones([v_size, v_size]) * np.inf
        previous_vertex = np.array([-1] * (v_size**2)).reshape(v_size, v_size)

        for (v1_id, v2_id), e in self.graph.edges.items():
            distance_matrix[v1_id - 1, v2_id - 1] = e.weight
            previous_vertex[v1_id - 1, v2_id - 1] = v1_id
        for id, v in self.graph.vertices.items():
            distance_matrix[id - 1, id - 1] = 0
            previous_vertex[id - 1, id - 1] = id

        for k in range(v_size):
            for i in range(v_size):
                for j in range(v_size):
                    if (
                        distance_matrix[i, j]
                        > distance_matrix[i, k] + distance_matrix[k, j]
                    ):
                        distance_matrix[i, j] = (
                            distance_matrix[i, k] + distance_matrix[k, j]
                        )
                        previous_vertex[i, j] = previous_vertex[k, j]

        return {
            id: min_dist
            for id, min_dist in zip(
                self.graph.vertices.keys(), distance_matrix[start_vertex - 1, :]
            )
        }

    def a_star_min_dist_algorithm(self, 
                                  start_vertex:int, goal_vertex:int, 
                                  heuristic) -> Optional[List[int]]:
        open_set = []
        heapq.heappush(open_set, (0, start_vertex))
        came_from = {start_vertex: None}
        g_score = {v.id: float("inf") for v in self.graph.vertices.values()}
        g_score[start_vertex] = 0
        f_score = {v.id: float("inf") for v in self.graph.vertices.values()}
        f_score[start_vertex] = heuristic(start_vertex, goal_vertex)

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_vertex:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                logger.info("Found solution using A*")
                return path[::-1]

            for neighbor in self.graph.vertices[current].leafs:
                edge = self.graph.edges[(current, neighbor)]
                tentative_g_score = g_score[current] + edge.weight
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_vertex)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None


class MinSpanningTreeAlgorithms:
    def __init__(self, G: Graph):
        """
        Input: an undirected, connected graph 𝐺 = (𝑉, 𝐸) with edge weights 𝑐 ∶ 𝐸 → Q.
        Task: Find a spanning tree 𝑇 in 𝐺 such that 𝑐(𝐸(𝑇)) = ∑_{𝑒∈𝐸(𝑇)} 𝑐(𝑒) is minimized

        Let (𝐺, 𝑐) be an instance of the minimum spanning tree problem with 𝐺 = (𝑉, 𝐸),
        and let 𝑇 = (𝑉, 𝐸_𝑇) be a spanning tree in 𝐺.
        Then the following statements are equivalent:

        (i) 𝑇 is a minimum spanning tree with respect to 𝑐.
        (ii) For every edge 𝑒 = {𝑢, 𝑣} ∈ 𝐸 ∖ 𝐸_𝑇, no edge on the 𝑢-𝑣-path in 𝑇 has a higher cost than 𝑐(𝑒).
        (iii) For every 𝑒 ∈ 𝐸_𝑇, and every connected component 𝐶 of 𝑇 − 𝑒 the cost 𝑐(𝑒) is minimum in 𝛿_𝐺(𝑉(𝐶)).
        (iv) We can order 𝐸_𝑇 = {𝑒_1, … , 𝑒_𝑛−1} such that for each 𝑖 ∈ {1, … , 𝑛 − 1} there exists a subset 𝑉′ ⊂ 𝑉
        of the nodes such that 𝑒_𝑖 is a minimum cost edge of 𝛿_𝐺(𝑉′) and 𝑒_𝑗 ∉ 𝛿_𝐺(𝑉′) for all 𝑗 ∈ {1, … , 𝑖 − 1}
        """

        self.graph = G

    def run(self,
            use_algorithm: MinSpanningTreeAlgorithmsEnum = MinSpanningTreeAlgorithmsEnum.KRUSKALS) -> Graph:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find minimum spanning tree")
        
            match use_algorithm.value:
                case 1: result = self.prims_min_spanning_tree_algorithm()
                case 2: result = self.kruskals_min_spanning_tree_algorithm()
                case _: logger.error("Algorithm doesn't exist returning no solution") 
        
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
        It can be implemented to run in time 𝒪(𝑛^2)"""

        v_0 = choice(tuple(self.graph.vertices.values()))
        T = Tree(Id="MST_Prims", V=[Vertex(v_0.id)], E=[])

        while len(T.vertices) < len(self.graph.vertices):
            edge_list = self.graph.edges.values()
            from_vertices = T.vertices.keys()
            to_vertices = [v for v in self.graph.vertices.keys() 
                           if v not in from_vertices]

            delta_edges = delta(edge_list, from_vertices, to_vertices)
            min_cost_edge = minimum_cost_edge_in_delta(delta_edges)
            if min_cost_edge is None: break
            T.add_edge(min_cost_edge.copy())

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

        Kruskal’s algorithm works correctly and can be implemented in time 𝒪(𝑚*𝑛)"""

        T = Tree(Id="MST_Kruskals", V=[], E=[])
        uf = UnionFind(self.graph.vertices.keys())
        edge_heap = [e.copy() for e in self.graph.edges.values()]
        heapq.heapify(edge_heap)

        while len(edge_heap) > 0:
            e = heapq.heappop(edge_heap)
            v1, v2 = e.incident_vertex_ids

            if uf.find(v1) != uf.find(v2):
                T.add_edge(e)
                uf.union(v1, v2)

        if len(T.vertices) < len(self.graph.vertices):
            logger.warning("Tree doesn't span the entirety of the Graph!!")

        return T


class NetworkFlowAlgorithms:
    def __init__(self, N: Network):
        self.network = N

    def run(self) -> Graph:
        pass


class MaxFlowAlgorithms(NetworkFlowAlgorithms):
    def __init__(self, N: Network):
        """
        Input: a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢)
        Task: compute an 𝑠-𝑡-flow 𝑓 in 𝑁 of maximum value

        An 𝑠-𝑡-flow 𝑓 in a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) is maximum if and only if there is no 𝑓-augmenting path

        Let 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) be a network, then the value of a maximum 𝑠-𝑡-flow
        is equal to the capacity of a minimum (𝑠, 𝑡)-cut in 𝑁"""

        super().__init__(N)

    def run(self,
            use_algorithm: MaxFlowAlgorithmsEnum = MaxFlowAlgorithmsEnum.EDMONDS_KARP) -> int:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find max flow / min cut")
        
            match use_algorithm.value:
                case 1: result = self.ford_fulkerson_max_flow_algorithm()
                case 2: result = self.edmonds_karp_max_flow_algorithm()
                case 3: result = self.dinics_max_flow_algorithm()
                case _: logger.error("Algorithm doesn't exist returning no solution") 
        
        return result

    def ford_fulkerson_max_flow_algorithm(self) -> float:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) with positive arc capacities 𝑢 ∶ 𝐸(𝐺) → Q>0
        output: an 𝑠-𝑡-flow of maximum value in 𝑁

        1 initialize 𝑓 as the zero flow, i. e., 𝑓(𝑒) ∶= 0 for all 𝑒 ∈ 𝐸(𝐺)
        2 while there exists an 𝑓-augmenting 𝑠-𝑡-path in 𝑁_𝑓 do
            3 compute an 𝑠-𝑡-path 𝑃 in 𝑁_𝑓
            4 set 𝛾 ∶= min{𝑢_𝑓(𝑒) ∶ 𝑒 ∈ 𝐸(𝑃)}
            5 augment 𝑓 along 𝑃 by 𝛾
        6 return 𝑓"""

        self.network.initialize_flow()
        augmentation_count = 0

        while True:
            path, flow = UtilAlgorithms.find_st_path(self.network)
            if path is None or flow == 0: break
            self.network.augment_along(path, flow)
            augmentation_count += 1

        logger.info(f"Found maximum flow in {augmentation_count} augmentation iterations using Ford-Fulkerson: {self.network.flow}")
        return self.network.flow

    def edmonds_karp_max_flow_algorithm(self) -> float:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) with positive arc capacities 𝑢 ∶ 𝐸(𝐺) → Q>0
        output: an 𝑠-𝑡-flow of maximum value in 𝑁

        1 initialize 𝑓 as the zero flow, i. e., 𝑓 (𝑒) ∶= 0 for all 𝑒 ∈ 𝐸(𝐺)
        2 while there exists an 𝑓-augmenting 𝑠-𝑡-path in 𝑁_𝑓 do
            3 compute an 𝑠-𝑡-path 𝑃 in 𝑁_𝑓 with a minimum number of edges
            4 set 𝛾 ∶= min{𝑢_𝑓(𝑒) ∶ 𝑒 ∈ 𝐸(𝑃)}
            5 augment 𝑓 along 𝑃 by 𝛾
        6 return 𝑓

        Regardless of the edge capacities, the Edmonds-Karp algorithm (Algorithm 5) stops after at most 𝑚*𝑛/2 augmentations.
        It can be implemented such that it computes a maximum network flow in time 𝒪(𝑚^2*𝑛)
        """

        self.network.initialize_flow()
        augmentation_count = 0

        while True:
            path, flow = UtilAlgorithms.find_st_path(self.network)
            if path is None or flow == 0: break
            self.network.augment_along(path, flow)
            augmentation_count += 1

        logger.info(f"Found maximum flow in {augmentation_count} augmentation iterations using Edmonds-Karp: {self.network.flow}")
        return self.network.flow

    def dinics_max_flow_algorithm(self) -> int:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢)
        output: a maximum 𝑠-𝑡 flow 𝑓 in 𝑁
        1 initialize 𝑓 ∶= 0
        2 while there exists an 𝑓-augmenting path in 𝑁𝑓 do
            3 initialize the layered residual network 𝑁^𝐿_𝑓
            4 while there exists an 𝑠-𝑡-path in 𝑁^𝐿_𝑓 do
                5 determine a node 𝑣 ∈ 𝑉(𝑁^𝐿_𝑓 ) of minimum throughput 𝑝(𝑣)
                6 determine flow augmentation 𝑓′ through PushFlow(𝑁^𝐿_𝑓 , 𝑣, 𝑝(𝑣))
                            and PullFlow(𝑁^𝐿_𝑓 , 𝑣, 𝑝(𝑣))
                7 update 𝑓 through augmenting by 𝑓′
                8 update 𝑁^𝐿_𝑓 : update capacities and throughput,
                                 remove nodes with throughput 0,
                                 remove arcs with capacity 0
            9 determine 𝑁_𝑓 with the current flow 𝑓
        10 return 𝑓"""

        self.network.initialize_flow()

        while self.network.update_node_levels():
            start = {v: 0 for v in self.network.vertices}
            while True:
                flow = UtilAlgorithms.blocking_flow(self.network, 
                                                    self.network.source_node_id, 
                                                    float('inf'), start)
                if flow == 0: break
                self.network.flow += flow 

        logger.info(f"Found maximum flow using Dinic's algorithm: {self.network.flow}")

        return self.network.flow
