import logging
from typing import Optional, List
from queue import PriorityQueue
import heapq
import numpy as np
from random import choice

from enums import MinDistanceAlgorithmsEnum, MinSpanningTreeAlgorithmsEnum
from graph_utils import delta, minimum_cost_edge_in_delta
from notation import Vertex, Edge, Graph, Tree


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
        self.graph = G

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
                case _: logger.error("Algorithm doesn't exist returning no solution"); return None
            return result

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

    def topological_sort_min_dist_algorithm(self, start_vertex:int) -> dict:
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
        distances = {v.id: float('inf') for v in self.graph.vertices.values()}
        distances[start_vertex] = 0
        for _ in range(len(self.graph.vertices) - 1):
            for edge in self.graph.edges.values():
                u, v = edge.end_vertex_ids
                if distances[u] + edge.weight < distances[v]:
                    distances[v] = distances[u] + edge.weight
        
        # Check for negative weight cycles
        for edge in self.graph.edges.values():
            u, v = edge.end_vertex_ids
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


    def a_star_min_dist_algorithm(self, start_vertex:int, goal_vertex:int, heuristic) -> Optional[dict]:
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
                while current:
                    path.append(current)
                    current = came_from[current]
                logger.info("Found solution using A Star")
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
        self.graph = G

    def run(self) -> Graph:
        pass

    def prims_min_spanning_tree_algorithm(self):
        v_0 = choice(self.graph.vertices)
        T = Tree(Id="MST_Prims", V=[v_0], E=[])

        while not T.vertices.issuperset(self.graph.vertices):
            delta_edges = delta(T.vertices, self)
            min_cost_edge = minimum_cost_edge_in_delta(delta)
            T.init_edge(min_cost_edge)
        
        return T


    def kruskals_min_spanning_tree_algorithm(self):
        T = Tree(Id="MST_Kruskals", V=[], E=[])
        pass


class NetworkFlowAlgorithms:
    def __init__(self, G: Graph):
        self.graph = G