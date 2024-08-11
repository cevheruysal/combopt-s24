import heapq
import logging
from collections import deque
from random import choice
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from enums import (GraphDirection, MaxFlowAlgorithmsEnum,
                   MinCostFlowAlgorithmsEnum, MinDistanceAlgorithmsEnum,
                   MinSpanningTreeAlgorithmsEnum)
from graph_utils import delta, minimum_cost_edge_in_delta
from notation import Arc, Edge, Graph, LinearProgram, Network, Tree, Vertex
from util_structs import UnionFind, VertexProp

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class UtilAlgorithms:
    """
    utility algorithms used throughout other algorithms 
    """
    @staticmethod
    def order_topologically(G:Graph) -> Optional[List[int]]:
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
    def find_uv_path(N: Network, 
                     init_vertex: int, end_vertex: int, 
                     init_flow: float = float("inf"),
                     leveled: bool = False) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    
        v_prop = VertexProp(N.vertices.keys())
        stack = [(init_vertex, None, init_flow)]
        visited = set()

        while stack:
            node, prev, flow = stack.pop()

            if node in visited: continue
            visited.add(node)
            v_prop.set_prev(node, prev)

            if node == end_vertex:
                path = v_prop.construct_path_to_node(init_vertex, end_vertex)
                logger.info(f"Found augmenting path: {path} with flow {flow}")
                return path, flow

            arcs = [N.edges[(node, leaf)] for leaf in N.vertices[node].leafs]
            arcs.sort(key=lambda arc: arc.remaining_capacity())

            for arc in arcs:
                _, leaf = arc.incident_vertex_ids
                
                if leaf not in visited and arc.remaining_capacity() > 0:
                    if leveled and N.node_levels[leaf] != N.node_levels[node] + 1:
                        continue
                    new_flow = min(flow, arc.remaining_capacity())
                    stack.append((leaf, node, new_flow))
                    
        return None, 0

    @staticmethod
    def find_st_path(N:Network) -> Optional[Tuple[List[Tuple[int, int]], float]]:
        return UtilAlgorithms.find_uv_path(N, N.source_node_id, N.sink_node_id)
    
    @staticmethod
    def min_throughput_node(N:Network) -> Tuple[Optional[int], float]:
        res_node: Optional[int] = None
        min_throughput: float = float('inf')
        
        for node in N.vertices:
            if (node == N.source_node_id or node == N.sink_node_id 
                or N.node_levels[node] == -1): 
                continue
            
            throughput_out, throughput_in = 0.0, 0.0

            for arc in N.edges.values():
                if arc.residual_arc: continue
                u,v = arc.incident_vertex_ids
                if u == node: throughput_out += arc.remaining_capacity()
                elif v == node: throughput_in += arc.remaining_capacity()
                
            node_throughput = min(throughput_out, throughput_in)
            if node_throughput > 0 and node_throughput < min_throughput:
                min_throughput = node_throughput
                res_node = node

        return res_node, min_throughput

    @staticmethod
    def push_pull_flow(N:Network, node:int, flow:int, direction:str) -> float:
        total_pushed = 0
        while flow > 0:
            if direction == "pull":           
                path, path_flow = UtilAlgorithms.find_uv_path(N, N.source_node_id, node, 
                                                              flow, leveled=True)
            elif direction == "push":
                path, path_flow = UtilAlgorithms.find_uv_path(N, node, N.sink_node_id, 
                                                              flow, leveled=True)
            else: ValueError("Wrong direction specified!")

            if path is None or path_flow == 0: break

            N.augment_along(path, path_flow, False)
            flow -= path_flow
            total_pushed += path_flow

        return total_pushed

    @staticmethod
    def blocking_flow(N:Network, node:int, throughput:int) -> int:
        pulled_flow = UtilAlgorithms.push_pull_flow(N, node, throughput, "pull")
        pushed_flow = UtilAlgorithms.push_pull_flow(N, node, pulled_flow, "push")

        if pulled_flow > pushed_flow:
            logger.error(f"Throughput flow not completely pushed, the result might be incorrect")
        
        return pushed_flow

    @staticmethod
    def push_relabel_init_flow(N:Network) -> None:
        """
        1 initialize 𝑓(e) := ⎧ 𝑢(𝑒), for 𝑒 ∈ 𝛿_out(𝑠)
                             ⎨ 0,    for 𝑒 ∈ 𝐸(𝐺) ∖ 𝛿_out(𝑠) """

        N.init_flow()
        s = N.source_node_id
        for v in N.vertices[s].leafs:
            arc = N.edges[(s, v)]
            N.augment_edge(s, v, arc.capacity)

    @staticmethod
    def push_relabel_init_phi(N:Network) -> Dict[int, int]:
        """
        2 initialize 𝜓(𝑣) ∶= ⎧ 𝑛,   for 𝑣 = 𝑠
                             ⎨ 0,   for 𝑣 ∈ 𝑉(𝐺) ∖ {𝑠} """
        
        return {v:len(N.vertices) if v == N.source_node_id else 0 for v in N.vertices}

    @staticmethod
    def push(N:Network, v:int, w:int) -> None:
        """
        Push(𝑁, 𝑓, 𝑒)
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢), a preflow 𝑓, an arc 𝑒 ∈ 𝐸(𝐺𝑓)
        output: a new preflow 𝑓 in 𝑁
        
        1 let (𝑣, 𝑤) ∶= 𝑒
        2 set 𝛾 ∶= min{ex_𝑓(𝑣), 𝑢_𝑓(𝑒)}
        3 augment 𝑓 along 𝑒 by 𝛾
        4 return 𝑓 """

        gamma = min(N.vertices[v].excess_flow(), 
                    N.edges[(v, w)].remaining_capacity())
        N.augment_edge(v, w, gamma)

    @staticmethod
    def relabel(N:Network, phi:Dict[int, int], v:int) -> None:
        """
        Relabel(𝑁, 𝑓, 𝜓, 𝑣)
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢), a preflow 𝑓, a distance labeling 𝜓, a node 𝑣 ∈ 𝑉(𝐺)
        output: a new distance labeling for 𝑁
        1 Set 𝜓(𝑣) ∶= min{𝜓(𝑤) + 1 ∶ (𝑣, 𝑤) ∈ 𝛿^out_𝐺_𝑓 (𝑣)}
        2 return 𝜓 """

        min_distance = float('inf')
        for w in N.vertices[v].leafs:
            if N.edges[(v, w)].remaining_capacity() > 0:
                min_distance = min(min_distance, phi[w])

        phi[v] = min_distance + 1

    def find_negative_cycle_dfs(G:Graph, 
                                parent:int, v:int, 
                                path_cost:float, 
                                current_path:List[Edge]) -> Optional[List[Edge]]:
        for leaf in G.vertices[v].leafs:
            edge = G.edges[(v, leaf)]
            path_cost += edge.weight
            current_path.append(edge)
                
            if leaf == parent and path_cost < 0:
                return current_path
            else:
                result = UtilAlgorithms.find_negative_cycle_dfs(G, parent, leaf, path_cost, current_path)
                if result is not None:
                    return None

    @staticmethod
    def find_negative_cycle(G:Graph) -> Optional[List[Edge]]:
        if not G.has_negative_w: return None

        negative_weighted_edges = [e for e in G.edges.values() if e.weight < 0]
        negative_weighted_edges.sort(key=lambda e: e.weight)
        for edge in negative_weighted_edges:
            u, v = edge.incident_vertex_ids
            _, path = UtilAlgorithms.find_negative_cycle_dfs(G, u, v, edge.weight, [edge])
            if path is not None:
                return path
        return None
    
    @staticmethod
    def init_b_flow(N:Network) -> None:
        edge_id = max((a.id for a in N.edges.values())) + 1
        s_id = max((v for v in N.vertices)) + 1
        t_id = s_id + 1

        added_edges:Set[Tuple[int, int]] = {}
        added_vertices = {s_id, t_id}

        for id, vertex in N.vertices.items():
            if id == s_id or id == t_id: continue
            if vertex.charge > 0:
                N.add_edge(Arc(edge_id, s_id, id, U=vertex.charge, R=False)); edge_id += 1
                N.add_edge(Arc(edge_id, id, s_id, R=True)); edge_id += 1
            
                added_edges.add((s_id, id))
                added_edges.add((id, s_id))
            
            elif vertex.charge < 0:
                N.add_edge(Arc(edge_id, id, t_id, U=-vertex.charge, R=False)); edge_id += 1
                N.add_edge(Arc(edge_id, t_id, id, R=True)); edge_id += 1

                added_edges.add((t_id, id))
                added_edges.add((id, t_id))
        
        N.update_meta()

        maxFlowInstance = MaxFlowAlgorithms(N)
        maxFlowInstance.run()

        for edge_id in added_edges:
            if edge_id in N.edges:
                N.del_edge(edge_id)

        for vertex_id in added_vertices:
            if vertex_id in N.vertices:
                N.del_vertex(vertex_id)
    
    def min_mean_cycle(N:Network) -> Optional[Tuple[List[Arc], float]]:
        """
        input : a digraph 𝐺 = (𝑉, 𝐸) with edge costs 𝑐 ∶ 𝐸 → Q
        output: a minimum mean cycle 𝐶∗ of mean cost 𝜇(𝐺) in 𝐺
        1 add a node 𝑠 to 𝐺
        2 add edges (𝑠, 𝑥) with cost 𝑐(𝑠, 𝑥) = 0 for all 𝑥 ∈ 𝑉
        3 set 𝐹_0(𝑠) = 0 and _𝐹0(𝑥) = ∞ for all 𝑥 ∈ 𝑉
        4 for 𝑘 = 1, … , 𝑛 do
            5 for 𝑥 ∈ 𝑉 do
                6 set _𝐹𝑘(𝑥) = ∞
                7 for (𝑤, 𝑥) ∈ 𝛿^in(𝑥) do
                    8 if _𝐹𝑘−1(𝑤) + 𝑐(𝑤, 𝑥) <_ 𝐹𝑘(𝑥) then
                    9 set _𝐹𝑘(𝑥) = _𝐹𝑘−1(𝑤) + 𝑐(𝑤, 𝑥)
        10 if _𝐹𝑛(𝑥) = ∞ for all 𝑥 ∈ 𝑉 then
            11 terminate, 𝐺 is acyclic
        12 compute  𝜇(𝐺) = mi n𝑥∈𝑉 max_{𝐹𝑛(𝑥) _− 𝐹𝑘( 𝑥)𝑛 −  𝑘∶ 0 ≤ 𝑘 ≤ 𝑛 − 1 ∶_ 𝐹𝑘(𝑥) < 
                     ∞}𝑥∗ = arg min𝑥∈𝑉 m_ax{𝐹𝑛(𝑥_) − 𝐹𝑘( 𝑥) 𝑛 − 𝑘∶ 0 ≤ 𝑘 ≤ 𝑛 − _1 ∶ 𝐹𝑘(𝑥) < ∞}
        13 let 𝐶∗ be the cycle on the edge progression corresponding to _𝐹𝑛(𝑥∗)
        14 return 𝐶∗ """

        # Step 1: Add a node s to G
        s = len(graph)
        graph[s] = {x: 0 for x in range(len(graph))}

        n = len(graph)
        
        # Step 3: Initialize F_0
        F = [[math.inf] * n for _ in range(n + 1)]
        F[0][s] = 0

        # Step 4: For k = 1, ..., n
        for k in range(1, n + 1):
            # Step 5: For each x in V
            for x in range(n):
                F[k][x] = math.inf
                # Step 7: For each (w, x) in δ^in(x)
                for w in range(n):
                    if w in graph and x in graph[w]:
                        # Step 8: Update F_k(x)
                        if F[k-1][w] + graph[w][x] < F[k][x]:
                            F[k][x] = F[k-1][w] + graph[w][x]

        # Step 10: Check if G is acyclic
        if all(F[n][x] == math.inf for x in range(n)):
            return None  # G is acyclic

        # Step 12: Compute μ(G) and find x*
        min_mu = math.inf
        x_star = None
        for x in range(n):
            max_mu_x = -math.inf
            for k in range(n):
                if F[k][x] < math.inf:
                    mu_xk = (F[n][x] - F[k][x]) / (n - k)
                    max_mu_x = max(max_mu_x, mu_xk)
            if max_mu_x < min_mu:
                min_mu = max_mu_x
                x_star = x

        # Step 13: Find the cycle corresponding to F_n(x*)
        C_star = []
        u = x_star
        k = n
        while k > 0:
            for w in range(n):
                if F[k-1][w] + graph[w][u] == F[k][u]:
                    C_star.append((w, u))
                    u = w
                    break
            k -= 1

        # Step 14: Return the minimum mean cycle
        return C_star[::-1], min_mu


class MinDistanceAlgorithms:
    def __init__(self, G:Graph, S:int, T:int,
                 H:Callable[[int, int], float] = lambda x, y: 1):
        self.graph = G
        self.start_vertex = S
        self.end_vertex = T
        self.heuristic = H

    def auto_run(self) -> Optional[dict]:
        result = None
        if self.graph.direction is GraphDirection.DIRECTED and self.graph.is_acyclical:
            logger.info("The graph is directed and acyclic, using Topological Sort to find minimum distance")
            result = self.topological_sort()
        elif self.graph.direction is GraphDirection.DIRECTED and not self.graph.has_negative_w:
            logger.info("The graph is directed and doesn't have negative weights, using Dijkstra's to find minimum distance")
            result = self.dijkstras()
        elif self.graph.has_negative_w:
            logger.info("The graph is either undirected or there is negative weighted edges, using Bellman-Ford's to find minimum distance")
            result = self.bellman_fords()
        else:
            logger.error("Automatic selection didn't run any minimum distance algorithm")
        if result is None:
            logger.error("Automatic selection couldn't find any solutions")
        return result

    def run(self, use_algorithm:MinDistanceAlgorithmsEnum = 0) -> Optional[dict]:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find minimum distance")

            match use_algorithm.value:
                case 1: result = self.topological_sort()
                case 2: result = self.dijkstras()
                case 3: result = self.bellman_fords()
                case 4: result = self.floyd_warshall()
                case 5: result = self.a_star() # TODO add heuristic selection scheme
                case _: logger.error("Algorithm doesn't exist returning no solution")
        else:
            result = self.auto_run()

        return result

    def topological_sort(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Let 𝐺 be an acyclic directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q and let 𝑠, 𝑡 ∈ 𝑉(𝐺).
        Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑛 + 𝑚)"""

        prop = VertexProp(self.graph.vertices.keys(),
                          self.start_vertex)

        for v_id in UtilAlgorithms.order_topologically(self.graph):
            if prop.get_dist(v_id) == float("inf"): continue

            for leaf in self.graph.vertices[v_id].leafs:
                edge = self.graph.edges[(v_id, leaf)]
                cand_dist_to_leaf = prop.get_dist(v_id) + edge.weight

                if cand_dist_to_leaf < prop.get_dist(leaf):
                    prop.set_dist(leaf, cand_dist_to_leaf)
                    prop.set_prev(leaf, v_id)

        logger.info("Found solution using Topological Sort")
        return (prop.get_dist(self.end_vertex),
                prop.construct_path_to_node(self.start_vertex, 
                                            self.end_vertex))

    def dijkstras(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Let 𝐺 be a directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q≥0 and let 𝑠, 𝑡 ∈ 𝑉(𝐺).
        Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑚 + 𝑛 log 𝑛)"""

        prop = VertexProp(self.graph.vertices.keys(),
                          self.start_vertex)
        
        heap = [(0, self.start_vertex)]
        heapq.heapify(heap)

        while heap:
            popped_dist, popped_v_id = heapq.heappop(heap)

            for leaf in self.graph.vertices[popped_v_id].leafs:
                edge = self.graph.edges[(popped_v_id, leaf)]
                cand_dist_to_leaf = popped_dist + edge.weight

                if cand_dist_to_leaf < prop.get_dist(leaf):
                    prop.set_dist(leaf, cand_dist_to_leaf)
                    prop.set_prev(leaf, popped_v_id)
                    
                    heapq.heappush(heap, (cand_dist_to_leaf, leaf))

        logger.info("Found solution using Dijkstra's")
        return (prop.get_dist(self.end_vertex),
                prop.construct_path_to_node(self.start_vertex, 
                                            self.end_vertex))

    def bellman_fords(self) -> Optional[Tuple[float, List[Tuple[int, int]]]:]:
        """
        Let 𝐺 = (𝑉, 𝐸) be a directed graph with edge weights 𝑐 ∶ 𝐸 → Q and let 𝑠, 𝑡 ∈ 𝑉.
        There is an algorithm that either computes a shortest 𝑠-𝑡-path in 𝐺
        or finds a negative cycle in 𝐺 in time 𝒪 (𝑚𝑛)"""

        prop = VertexProp(self.graph.vertices.keys(),
                          self.start_vertex)

        # for every possible k step path starting from s compute dist 0<k<n
        for _ in range(len(self.graph.vertices) - 1):
            for edge in self.graph.edges.values():
                u, v = edge.incident_vertex_ids
                cand_dist_to_v = prop.get_dist(u) + edge.weight
                if cand_dist_to_v < prop.get_dist(v):
                    prop.set_dist(v, cand_dist_to_v)
                    prop.set_prev(v, u)

        # if a dist to any vertex decreases when we perform the nth step return cycle!
        for edge in self.graph.edges.values():
            u, v = edge.incident_vertex_ids
            if prop.get_dist(u) + edge.weight < prop.get_dist(v):
                logger.info("Graph contains a negative weight cycle")
                return None

        logger.info("Found solution using Bellman Ford's")
        return (prop.get_dist(self.end_vertex),
                prop.construct_path_to_node(self.start_vertex, 
                                            self.end_vertex))
    
    def floyd_warshall(self) -> Tuple[float, List[Tuple[int, int]]]:
        props = {v: VertexProp(self.graph.vertices.keys()) for v in self.graph.vertices}

        for (v1_id, v2_id), e in self.graph.edges.items():
            props[v1_id].set_dist(v2_id, e.weight)
            props[v1_id].set_prev(v2_id, v1_id)

        for v_id in self.graph.vertices:
            props[v_id].set_dist(v_id, 0)
            props[v_id].set_prev(v_id, v_id)

        for k in self.graph.vertices:
            for i in self.graph.vertices:
                for j in self.graph.vertices:
                    if props[i].get_dist(j) > props[i].get_dist(k) + props[k].get_dist(j):
                        props[i].set_dist(j, props[i].get_dist(k) + props[k].get_dist(j))
                        props[i].set_prev(j, props[k].get_prev(j))

        return (props[self.start_vertex].get_dist(self.end_vertex),
                props[self.start_vertex].construct_path_to_node(self.start_vertex, 
                                                                self.end_vertex))

    """def floyd_warshall(self) -> Tuple[float, List[int]]:
        v_size = len(self.graph.vertices)
        distance_matrix = np.full((v_size, v_size), np.inf)
        np.fill_diagonal(distance_matrix, 0)
        previous_vertex = np.full((v_size, v_size), -1)
        
        for (v1_id, v2_id), e in self.graph.edges.items():
            distance_matrix[v1_id - 1, v2_id - 1] = e.weight
            previous_vertex[v1_id - 1, v2_id - 1] = v1_id - 1
        
        for k in range(v_size):
            i_k_dist = distance_matrix[:, k].reshape(-1, 1)
            k_j_dist = distance_matrix[k, :].reshape(1, -1)
            new_dist = i_k_dist + k_j_dist
            
            mask = distance_matrix > new_dist
            distance_matrix[mask] = new_dist[mask]
            previous_vertex[mask] = previous_vertex[k, mask.argmax(axis=0)]
        
        start_idx = self.start_vertex - 1
        end_idx = self.end_vertex - 1
        min_distance = distance_matrix[self.start_vertex - 1, 
                                       self.end_vertex - 1]
        
        path = []
        if np.isfinite(min_distance):
            current_vertex = end_idx
            while current_vertex != start_idx:
                path.append((previous_vertex[start_idx, current_vertex], 
                             current_vertex + 1))
                current_vertex = previous_vertex[start_idx, current_vertex]
                if current_vertex == -1:
                    path = []
                    break
            path.reverse()
        
        return min_distance, path"""

    def a_star(self) -> Optional[List[int]]:
        open_set = []
        heapq.heappush(open_set, (0, self.start_vertex))
        came_from = {self.start_vertex: None}
        g_score = {v: 0 if v == self.start_vertex 
                        else float("inf") for v in self.graph.vertices}
        f_score = {v: self.heuristic(self.start_vertex, self.end_vertex) if v == self.start_vertex 
                        else float("inf") for v in self.graph.vertices}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == self.end_vertex:
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
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.end_vertex)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None


class MinSpanningTreeAlgorithms:
    def __init__(self, G:Graph):
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

    def run(self, use_algorithm:MinSpanningTreeAlgorithmsEnum = 2) -> Graph:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find minimum spanning tree")

            match use_algorithm.value:
                case 1: result = self.prims()
                case 2: result = self.kruskals()
                case _: logger.error("Algorithm doesn't exist returning no solution")

        return result

    def prims(self):
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

        while True:
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

    def kruskals(self):
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

        while edge_heap:
            e = heapq.heappop(edge_heap)
            v1, v2 = e.incident_vertex_ids

            if uf.find(v1) != uf.find(v2):
                T.add_edge(e)
                uf.union(v1, v2)

        if len(T.vertices) < len(self.graph.vertices):
            logger.warning("Tree doesn't span the entirety of the Graph!!")

        return T


class MaxFlowAlgorithms():
    def __init__(self, N:Network):
        """
        Input: a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢)
        Task: compute an 𝑠-𝑡-flow 𝑓 in 𝑁 of maximum value

        An 𝑠-𝑡-flow 𝑓 in a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) is maximum if and only if there is no 𝑓-augmenting path

        Let 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) be a network, then the value of a maximum 𝑠-𝑡-flow
        is equal to the capacity of a minimum (𝑠, 𝑡)-cut in 𝑁"""

        self.network = N


    def run(self, use_algorithm:MaxFlowAlgorithmsEnum = 2) -> Optional[int]:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find max flow / min cut")

            match use_algorithm.value:
                case 1: result = self.ford_fulkerson()
                case 2: result = self.edmonds_karp()
                case 3: result = self.dinics()
                case 4: result = self.push_relabel()
                case _: logger.error("Algorithm doesn't exist returning no solution")

        return result

    def ford_fulkerson(self) -> float:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) with positive arc capacities 𝑢 ∶ 𝐸(𝐺) → Q>0
        output: an 𝑠-𝑡-flow of maximum value in 𝑁

        1 initialize 𝑓 as the zero flow, i. e., 𝑓(𝑒) ∶= 0 for all 𝑒 ∈ 𝐸(𝐺)
        2 while there exists an 𝑓-augmenting 𝑠-𝑡-path in 𝑁_𝑓 do
            3 compute an 𝑠-𝑡-path 𝑃 in 𝑁_𝑓
            4 set 𝛾 ∶= min{𝑢_𝑓(𝑒) ∶ 𝑒 ∈ 𝐸(𝑃)}
            5 augment 𝑓 along 𝑃 by 𝛾
        6 return 𝑓"""

        self.network.init_flow()
        augmentation_count = 0

        while True:
            path, flow = UtilAlgorithms.find_st_path(self.network)
            if path is None or flow == 0: break
            self.network.augment_along(path, flow)
            augmentation_count += 1

        logger.info(f"Found maximum flow in {augmentation_count} augmentation iterations using Ford-Fulkerson: {self.network.flow}")
        return self.network.flow

    def edmonds_karp(self) -> float:
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

        self.network.init_flow()
        augmentation_count = 0

        while True:
            path, flow = UtilAlgorithms.find_st_path(self.network)
            if path is None or flow == 0: break
            self.network.augment_along(path, flow)
            augmentation_count += 1

        logger.info(f"Found maximum flow in {augmentation_count} augmentation iterations using Edmonds-Karp: {self.network.flow}")
        return self.network.flow

    def dinics(self) -> int:
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

        self.network.init_flow()

        while self.network.update_node_levels():
            node, throughput = UtilAlgorithms.min_throughput_node(self.network)
            if node is None: break
            flow = UtilAlgorithms.blocking_flow(self.network, node, throughput)
            if flow == 0: break
            self.network.flow += flow

        logger.info(f"Found maximum flow using Dinic's algorithm: {self.network.flow}")
        return self.network.flow

    def push_relabel(self) -> int:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢)
        output: a maximum 𝑠-𝑡 flow 𝑓 in 𝑁

        1 initialize 𝑓(e) := ⎧ 𝑢(𝑒), for 𝑒 ∈ 𝛿_out(𝑠)
                             ⎨ 0,    for 𝑒 ∈ 𝐸(𝐺) ∖ 𝛿_out(𝑠)
        
        2 initialize 𝜓(𝑣) ∶= ⎧ 𝑛,   for 𝑣 = 𝑠
                             ⎨ 0,   for 𝑣 ∈ 𝑉(𝐺) ∖ {𝑠}
        
        3 while there exists an active node 𝑣 with ex_𝑓(𝑣) > 0 do
            4 select an active node 𝑣 such that 𝜓(𝑣) = max{𝜓(𝑤) ∶ 𝑤 ∈ 𝑉(𝐺), 𝑤 is active}
            5 if there exists an admissible edge 𝑒 ∈ 𝛿^out_𝐺_𝑓 (𝑣) then
                6 Push(𝑁, 𝑓, 𝑒)
            7 else
                8 Relabel(𝑁, 𝑓, 𝜓, 𝑣)
        9 return 𝑓 """

        UtilAlgorithms.push_relabel_init_flow(self.network)
        phi:Dict[int, int] = UtilAlgorithms.push_relabel_init_phi(self.network)

        active_nodes = [v for v in self.network.vertices.values()
                        if v.id != self.network.source_node_id
                       and v.id != self.network.sink_node_id
                       and v.excess_flow() > 0]

        while active_nodes:
            pushed = False
            node = max(active_nodes, key=lambda x: phi[x.id])
            
            for w in node.leafs:
                if (self.network.edges[(node.id, w)].remaining_capacity() > 0 and 
                    phi[node.id] == phi[w] + 1):
                    
                    pushed = True
                    UtilAlgorithms.push(self.network, node.id, w)

                    if node.excess_flow() == 0: active_nodes.remove(node)
                    leaf = self.network.vertices[w]
                    if (leaf.id != self.network.source_node_id
                        and leaf.id != self.network.sink_node_id
                        and leaf.excess_flow() > 0 
                        and leaf not in active_nodes):
                        active_nodes.append(leaf)
                        
                    break

            if not pushed:
                UtilAlgorithms.relabel(self.network, phi, node.id)

        self.network.flow = self.network.vertices[self.network.sink_node_id].excess_flow()

        logger.info(f"Found maximum flow using Push-Relabel algorithm: {self.network.flow}")
        return self.network.flow


class MinCostFlowAlgorithms():
    def __init__(self, N:Network):
        """
         """

        self.network = N

    def run(self, use_algorithm:MinCostFlowAlgorithmsEnum = 0) -> int:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find min cost flow")

            match use_algorithm.value:
                case 1: result = self.minimum_mean_cycle_cancelling()
                case 2: result = self.successive_shortest_path()
                case _: logger.error("Algorithm doesn't exist returning no solution")

        return result

    def minimum_mean_cycle_cancelling(self):
        """
        input : a minimum cost flow instance 𝑁 = (𝐺, 𝑢, 𝑐, 𝑏)
        output: a minimum cost 𝑏-flow for the given instance
        1 initialize 𝑓 as any 𝑏-flow (or stop if no 𝑏-flow exists) using a maxflow algorithm
        2 while there exists a cycle of negative cost in 𝐺_𝑓 do
            3 determine a cycle 𝐶 in 𝐺_𝑓 that minimizes 𝑐_𝑓(𝐶)/|𝐶| < 0
            4 augment 𝑓 along 𝐶 to obtain the new 𝑏-flow 𝑓 ∶= 𝑓_𝐶
        5 return 𝑓 """

        if not UtilAlgorithms.init_b_flow(self.network):
            logger.error("No feasible b-flow exists returning no solution")
            return None

        while True:
            cycle = UtilAlgorithms.find_negative_cycle(self.network)
            if cycle is None: break
            min_capacity = min(self.network.edges[edge].remaining_capacity() for edge in cycle)
            self.network.augment_along(cycle, min_capacity)

        return sum(arc.flow * arc.weight for arc in self.network.edges.values())

    def successive_shortest_path(self):
        while True:
            pred = MinDistanceAlgorithms.dijkstras(self.network, 
                                                   self.network.source_node_id, 
                                                   self.network.sink_node_id)
            if pred[self.network.sink_node_id] is None:
                break

            v = self.network.sink_node_id
            min_capacity = float('inf')
            while pred[v] is not None:
                u = pred[v]
                min_capacity = min(min_capacity, self.network.edges[(u, v)].remaining_capacity())
                v = u

            v = self.network.sink_node_id
            while pred[v] is not None:
                u = pred[v]
                self.network.augment_edge(u, v, min_capacity)
                v = u

        return sum(arc.flow for arc in self.network.edges.values() 
                   if arc.incident_vertex_ids[0] == self.network.source_node_id)


class LinearOptimizationAlgorithms:
    def __init__(self, lp, tol=1e-6, max_iter=1_000_000):
        self.lp: LinearProgram = lp
        self.tol: float = tol
        self.max_iter: int = max_iter

    def ellipsoid_method(self):
        n = len(self.lp.c)
        x = np.zeros(n)
        Q = np.eye(n) * 100
        for _ in range(self.max_iter):
            if np.dot(self.lp.c, x) - np.dot(self.lp.c, self.lp.b) < self.tol:
                return x
            violated_constraint_index = np.argmax(np.dot(self.lp.A, x) - self.lp.b)
            a = self.lp.A[violated_constraint_index]
            alpha = np.dot(a, x) - self.lp.b[violated_constraint_index]
            if alpha <= 0:
                return x
            a_norm = np.linalg.norm(a)
            x = x - (alpha / a_norm**2) * np.dot(Q, a)
            Q = (n**2 / (n**2 - 1)) * (Q - (2 / (n + 1)) * np.outer(np.dot(Q, a), np.dot(Q, a)) / np.dot(a, np.dot(Q, a)))
        return x

    def simplex_method(self):
        c = self.lp.c
        A = self.lp.A
        b = self.lp.b
        
        # Initialize basic and non-basic variables
        m, n = A.shape
        basic = list(range(n, n + m))
        non_basic = list(range(n))
        
        # Create the initial tableau
        tableau = np.hstack([A, np.eye(m), b.reshape(-1, 1)])
        tableau = np.vstack([np.hstack([c, np.zeros(m + 1)]), tableau])
        
        while True:
            # Check for optimality
            if all(tableau[0, :-1] >= 0):
                solution = np.zeros(n)
                solution[basic] = tableau[1:, -1]
                return solution
            
            # Determine entering variable (most negative cost coefficient)
            entering = np.argmin(tableau[0, :-1])
            
            # Determine leaving variable
            ratios = tableau[1:, -1] / tableau[1:, entering]
            ratios[tableau[1:, entering] <= 0] = np.inf
            leaving = np.argmin(ratios) + 1
            
            # Pivot
            pivot = tableau[leaving, entering]
            tableau[leaving, :] /= pivot
            for i in range(tableau.shape[0]):
                if i != leaving:
                    tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]
            
            # Update basic and non-basic variables
            basic[leaving - 1], non_basic[entering] = non_basic[entering], basic[leaving - 1]

    def interior_point_method(self):
        A = self.lp.A
        b = self.lp.b
        c = self.lp.c
        
        # Initialize x, s, and lambda
        n = len(c)
        x = np.ones(n)
        s = np.ones(len(b))
        lmbda = np.ones(len(b))
        
        def F(x, s, lmbda):
            return np.hstack([A.T @ lmbda - c,
                              A @ x - s - b,
                              np.diag(s) @ np.diag(lmbda) @ np.ones(len(b))])
        
        for _ in range(self.max_iter):
            r = F(x, s, lmbda)
            if np.linalg.norm(r) < self.tol:
                return x
            
            # Solve the linear system using a basic Newton method
            J = np.block([[np.zeros((n, n)), np.zeros((n, len(b))), A.T],
                          [A, -np.eye(len(b)), np.zeros((len(b), len(b)))],
                          [np.zeros((len(b), n)), np.diag(lmbda), np.diag(s)]])
            
            delta = np.linalg.solve(J, -r)
            dx = delta[:n]
            ds = delta[n:n+len(b)]
            dlmbda = delta[n+len(b):]
            
            # Update variables with a step size
            step_size = min(1, 0.9 * min(-s[ds < 0] / ds[ds < 0], -lmbda[dlmbda < 0] / dlmbda[dlmbda < 0], default=1))
            x += step_size * dx
            s += step_size * ds
            lmbda += step_size * dlmbda
        
        return x