import logging
from typing import Optional, List
from queue import PriorityQueue
import heapq
import numpy as np
from random import choice

from enums import MaxFlowAlgorithmsEnum, MinDistanceAlgorithmsEnum, MinSpanningTreeAlgorithmsEnum
from graph_utils import delta, minimum_cost_edge_in_delta
from notation import Vertex, Edge, Graph, Tree
from util_structs import UnionFind


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class UtilAlgorithms:
    @staticmethod
    def topological_sort(G:Graph) -> Optional[List[int]]:
        if len(G.edges) == 0:
            logger.warning("No edges present to perform topological sorting")
            return None
        
        topological_order = []
        no_root_vertices_list = [v.copy() for v in G.vertices.values() if len(v.roots) == 0]
        all_vertices_dict = {id: v.copy() for id, v in G.vertices.items() if len(v.roots) > 0}

        while no_root_vertices_list:
            root_vertex = no_root_vertices_list.pop(0)
            topological_order.append(root_vertex.id)
            
            for leaf_id in root_vertex.leafs:
                leaf_vertex = all_vertices_dict[leaf_id]
                leaf_vertex.roots.remove(root_vertex.id)
                
                if len(leaf_vertex.roots) == 0:
                    no_root_vertices_list.append(leaf_vertex.copy())

        if len(topological_order) == len(G.vertices):
            return topological_order
        
        logger.info("Topological sorting finished, no possible sorting found. The graph may have cycles.")
        return None


class MinDistanceAlgorithms:
    def __init__(self, G: Graph):
        """
        Input: an undirected, connected graph 𝐺 = (𝑉, 𝐸) with edge weights 𝑐 ∶ 𝐸 → Q.
        Task: Find a spanning tree 𝑇 in 𝐺 such that 𝑐(𝐸(𝑇)) = ∑_{𝑒∈𝐸(𝑇)} 𝑐(𝑒) is minimized """

        self.graph = G

    def auto_run(self, start_vertex:int, goal_vertex:int) -> Optional[dict]:
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

    def run(self, start_vertex:int, goal_vertex:int, 
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

    def topological_sort_min_dist_algorithm(self, start_vertex:int) -> dict:
        """
        Let 𝐺 be an acyclic directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q and let 𝑠, 𝑡 ∈ 𝑉(𝐺). 
        Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑛 + 𝑚) """

        distances = {v.id: float('inf') for v in self.graph.vertices.values()}
        distances[start_vertex] = 0
        for vertex_id in UtilAlgorithms.topological_sort(self.graph):
            if distances[vertex_id] == float('inf'):
                continue
            for neighbor in self.graph.vertices[vertex_id].leafs:
                edge = self.graph.edges[(vertex_id, neighbor)]
                new_distance = distances[vertex_id] + edge.weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
        logger.info("Found solution using Topological Sort")
        return distances

    def dijkstras_min_dist_algorithm(self, start_vertex:int) -> dict:
        """
        Let 𝐺 be a directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q≥0 and let 𝑠, 𝑡 ∈ 𝑉(𝐺). 
        Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑚 + 𝑛 log 𝑛) """

        distances = {v.id: float('inf') for v in self.graph.vertices.values()}
        distances[start_vertex] = 0
        pq = PriorityQueue()
        pq.put((0, start_vertex))
        
        while not pq.empty():
            current_distance, current_vertex = pq.get()
            if current_distance > distances[current_vertex]:
                continue

            for neighbor in self.graph.vertices[current_vertex].leafs:
                edge = self.graph.edges[(current_vertex, neighbor)]
                distance = current_distance + edge.weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    pq.put((distance, neighbor))
        logger.info("Found solution using Dijkstra's")
        return distances

    def bellman_fords_min_dist_algorithm(self, start_vertex:int) -> Optional[dict]:
        """
        Let 𝐺 = (𝑉, 𝐸) be a directed graph with edge weights 𝑐 ∶ 𝐸 → Q and let 𝑠, 𝑡 ∈ 𝑉. 
        There is an algorithm that either computes a shortest 𝑠-𝑡-path in 𝐺 
        or finds a negative cycle in 𝐺 in time 𝒪 (𝑚𝑛) """
        
        distances = {v.id: float('inf') for v in self.graph.vertices.values()}
        distances[start_vertex] = 0
        for _ in range(len(self.graph.vertices) - 1):
            for edge in self.graph.edges.values():
                u, v = edge.incident_vertex_ids
                if distances[u] + edge.weight < distances[v]:
                    distances[v] = distances[u] + edge.weight
        
        # Check for negative weight cycles
        for edge in self.graph.edges.values():
            u, v = edge.incident_vertex_ids
            if distances[u] + edge.weight < distances[v]:
                raise logger.info("Graph contains a negative weight cycle")
                return None
        logger.info("Found solution using Bellman Ford's")
        return distances
    
    def floyd_warshall_min_dist_algorithm(self, start_vertex:int) -> Optional[dict]:
        v_size = len(self.graph.vertices)
        distance_matrix = np.ones([v_size, v_size]) * np.inf
        previous_vertex = np.array([-1]*(v_size**2)).reshape(v_size, v_size)

        for (v1_id,v2_id), e in self.graph.edges.items():
            distance_matrix[v1_id-1, v2_id-1] = e.weight
            previous_vertex[v1_id-1, v2_id-1] = v1_id
        for id, v in self.graph.vertices.items():
            distance_matrix[id-1, id-1] = 0
            previous_vertex[id-1, id-1] = id
        
        for k in range(v_size):
            for i in range(v_size):
                for j in range(v_size):
                    if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
                        previous_vertex[i, j] = previous_vertex[k, j]
        
        return {id: min_dist for id, min_dist in zip(self.graph.vertices.keys() ,distance_matrix[start_vertex-1, :])}

    def a_star_min_dist_algorithm(self, start_vertex:int, goal_vertex:int, heuristic) -> Optional[List[int]]:
        open_set = []
        heapq.heappush(open_set, (0, start_vertex))
        came_from = {start_vertex: None}
        g_score = {v.id: float('inf') for v in self.graph.vertices.values()}
        g_score[start_vertex] = 0
        f_score = {v.id: float('inf') for v in self.graph.vertices.values()}
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
        Let (𝐺, 𝑐) be an instance of the minimum spanning tree problem with 𝐺 = (𝑉, 𝐸), 
        and let 𝑇 = (𝑉, 𝐸_𝑇) be a spanning tree in 𝐺. 
        Then the following statements are equivalent:

        (i) 𝑇 is a minimum spanning tree with respect to 𝑐.
        (ii) For every edge 𝑒 = {𝑢, 𝑣} ∈ 𝐸 ∖ 𝐸_𝑇, no edge on the 𝑢-𝑣-path in 𝑇 has a higher cost than 𝑐(𝑒).
        (iii) For every 𝑒 ∈ 𝐸_𝑇, and every connected component 𝐶 of 𝑇 − 𝑒 the cost 𝑐(𝑒) is minimum in 𝛿_𝐺(𝑉(𝐶)).
        (iv) We can order 𝐸_𝑇 = {𝑒_1, … , 𝑒_𝑛−1} such that for each 𝑖 ∈ {1, … , 𝑛 − 1} there exists a subset 𝑉′ ⊂ 𝑉
        of the nodes such that 𝑒_𝑖 is a minimum cost edge of 𝛿_𝐺(𝑉′) and 𝑒_𝑗 ∉ 𝛿_𝐺(𝑉′) for all 𝑗 ∈ {1, … , 𝑖 − 1} """

        self.graph = G

    def run(self, use_algorithm:MinSpanningTreeAlgorithmsEnum = MinSpanningTreeAlgorithmsEnum.KRUSKALS) -> Graph:
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
        It can be implemented to run in time 𝒪(𝑛^2) """

        v_0 = choice(self.graph.vertices)
        T = Tree(Id="MST_Prims", V=[v_0], E=[])

        while not T.vertices.issuperset(self.graph.vertices):
            edge_list, from_vertices, to_vertices = self.graph.edges, T.vertices, self.graph.vertices
            delta_edges = delta(edge_list, from_vertices, to_vertices)
            min_cost_edge = minimum_cost_edge_in_delta(delta_edges)
            if min_cost_edge is None: break
            T.add_edge(min_cost_edge)
        
        if not T.vertices.issuperset(self.graph.vertices):
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

        if not T.vertices.issuperset(self.graph.vertices):
            logger.warning("Tree doesn't span the entirety of the Graph!!")
            
        return T


class NetworkFlowAlgorithms:
    def __init__(self, G: Graph):
        self.graph = G

    def run(self) -> Graph:
        pass

class MaxFlowAlgorithms(NetworkFlowAlgorithms):
    def __init__(self, G: Graph):
        """
        Input: a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢)
        Task: compute an 𝑠-𝑡-flow 𝑓 in 𝑁 of maximum value 
        
        An 𝑠-𝑡-flow 𝑓 in a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) is maximum if and only if there is no 𝑓-augmenting path 
        
        Let 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) be a network, then the value of a maximum 𝑠-𝑡-flow 
        is equal to the capacity of a minimum (𝑠, 𝑡)-cut in 𝑁 """

        super().__init__(G)
        self.residual_graph = None

    def run(self, source_vertex_id:Optional[int]=0, sink_vertex_id:Optional[int]=None, 
            use_algorithm:MaxFlowAlgorithmsEnum=MaxFlowAlgorithmsEnum.EDMONDS_KARP) -> int:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find max flow / min cut")
        
            match use_algorithm.value:
                case 1: result = self.ford_fulkerson_max_flow_algorithm(source_vertex_id, sink_vertex_id)
                case 2: result = self.edmonds_karp_max_flow_algorithm(source_vertex_id, sink_vertex_id)
                case 3: result = self.dinics_max_flow_algorithm(source_vertex_id, sink_vertex_id)
                case _: logger.error("Algorithm doesn't exist returning no solution") 
        
        return result

    def ford_fulkerson_max_flow_algorithm(self, source:Optional[int]=0, sink:Optional[int]=None) -> int:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) with positive arc capacities 𝑢 ∶ 𝐸(𝐺) → Q>0
        output: an 𝑠-𝑡-flow of maximum value in 𝑁
        
        1 initialize 𝑓 as the zero flow, i. e., 𝑓(𝑒) ∶= 0 for all 𝑒 ∈ 𝐸(𝐺)
        2 while there exists an 𝑓-augmenting 𝑠-𝑡-path in 𝑁_𝑓 do
            3 compute an 𝑠-𝑡-path 𝑃 in 𝑁_𝑓 
            4 set 𝛾 ∶= min{𝑢_𝑓(𝑒) ∶ 𝑒 ∈ 𝐸(𝑃)}
            5 augment 𝑓 along 𝑃 by 𝛾
        6 return 𝑓 """

        if sink is None: sink = len(self.graph.vertices) - 1 
        max_flow = 0
        self.residual_graph = self._create_residual_graph()

        while True:
            path, flow = self._bfs(source, sink)
            if not path:
                break
            max_flow += flow
            self._update_residual_graph(path, flow)

        logger.info(f"Found maximum flow using Ford-Fulkerson: {max_flow}")
        return max_flow

    def edmonds_karp_max_flow_algorithm(self, source:Optional[int]=0, sink:Optional[int]=None) -> int:
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
        It can be implemented such that it computes a maximum network flow in time 𝒪(𝑚^2*𝑛) """

        if sink is None: sink = len(self.graph.vertices) - 1 
        max_flow = 0
        self.residual_graph = self._create_residual_graph()

        while True:
            path, flow = self._bfs(source, sink)
            if not path:
                break
            max_flow += flow
            self._update_residual_graph(path, flow)

        logger.info(f"Found maximum flow using Edmonds-Karp: {max_flow}")
        return max_flow

    def dinics_max_flow_algorithm(self, source:Optional[int]=0, sink:Optional[int]=None) -> int:
        if sink is None: sink = len(self.graph.vertices) - 1 
        max_flow = 0
        self.residual_graph = self._create_residual_graph()

        while self._bfs_level_graph(source, sink):
            start = [0] * len(self.graph.vertices)
            while True:
                flow = self._dfs_blocking_flow(source, sink, float('inf'), start)
                if flow == 0:
                    break
                max_flow += flow

        logger.info(f"Found maximum flow using Dinic's algorithm: {max_flow}")
        return max_flow

    def _create_residual_graph(self):
        residual_graph = {}
        for u in self.graph.vertices:
            residual_graph[u] = {}
            for v in self.graph.vertices:
                residual_graph[u][v] = 0
        for edge in self.graph.edges.values():
            u, v = edge.incident_vertex_ids
            residual_graph[u][v] = edge.weight
        return residual_graph

    def _bfs(self, source, sink):
        parent = {v: -1 for v in self.graph.vertices}
        visited = set()
        queue = [(source, float('inf'))]
        
        while queue:
            current, flow = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.graph.vertices[current].leafs:
                if neighbor not in visited and self.residual_graph[current][neighbor] > 0:
                    new_flow = min(flow, self.residual_graph[current][neighbor])
                    parent[neighbor] = current
                    if neighbor == sink:
                        path = self._construct_path(parent, sink)
                        return path, new_flow
                    queue.append((neighbor, new_flow))
        return None, 0

    def _construct_path(self, parent, sink):
        path = []
        current = sink
        while parent[current] != -1:
            path.insert(0, (parent[current], current))
            current = parent[current]
        return path

    def _update_residual_graph(self, path, flow):
        for u, v in path:
            self.residual_graph[u][v] -= flow
            self.residual_graph[v][u] += flow

    def _bfs_level_graph(self, source, sink):
        level = {v: -1 for v in self.graph.vertices}
        level[source] = 0
        queue = [source]
        
        while queue:
            u = queue.pop(0)
            for v in self.graph.vertices[u].leafs:
                if level[v] < 0 and self.residual_graph[u][v] > 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        
        self.level = level
        return level[sink] != -1

    def _dfs_blocking_flow(self, u, sink, flow, start):
        if u == sink:
            return flow
        while start[u] < len(self.graph.vertices[u].leafs):
            v = self.graph.vertices[u].leafs[start[u]]
            if self.level[v] == self.level[u] + 1 and self.residual_graph[u][v] > 0:
                curr_flow = min(flow, self.residual_graph[u][v])
                temp_flow = self._dfs_blocking_flow(v, sink, curr_flow, start)
                if temp_flow > 0:
                    self.residual_graph[u][v] -= temp_flow
                    self.residual_graph[v][u] += temp_flow
                    return temp_flow
            start[u] += 1
        return 0
