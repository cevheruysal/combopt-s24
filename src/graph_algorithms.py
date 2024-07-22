import logging
from typing import Optional, List
from queue import PriorityQueue
import heapq
import numpy as np
from random import choice

from enums import MinDistanceAlgorithmsEnum, MinSpanningTreeAlgorithmsEnum
from graph_utils import delta
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
        
        sorted_vertices = []
        all_vertices_dict = {id: v.copy() for id, v in G.vertices.items()}
        no_root_vertices_list = [v.copy() for v in G.vertices.values() if len(v.roots) == 0]
        remaining_edges_dict = {(e.end_vertex_ids[0], e.end_vertex_ids[1]): 1 for e in G.edges.values()}

        while no_root_vertices_list:
            root_vertex = no_root_vertices_list.pop(0)
            sorted_vertices.append(root_vertex.id)
            
            for leaf_id in root_vertex.leafs:
                leaf_vertex = all_vertices_dict[leaf_id]
                leaf_vertex.roots.remove(root_vertex.id)
                
                remaining_edges_dict[(root_vertex.id, leaf_id)] = 0
                
                if len(leaf_vertex.roots) == 0:
                    no_root_vertices_list.append(leaf_vertex.copy())

        if any(remaining_edges_dict.values()):
            logger.info("Topological sorting finished no possible sorting found the graph is acyclical")
            return None
        
        return sorted_vertices

class MinDistanceAlgorithms:
    def __init__(self, G: Graph):
        self.graph = G

    def run(self, start_vertex:int, goal_vertex:int, 
            use_algorithm:MinDistanceAlgorithmsEnum=MinDistanceAlgorithmsEnum.AUTOMATIC) -> Optional[dict]:
        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find minimum distance")
            match use_algorithm.value:
                case 1: return self.topological_sort_min_dist_algorithm(start_vertex)
                case 2: return self.dijkstras_min_dist_algorithm(start_vertex)
                case 3: return self.bellman_fords_min_dist_algorithm(start_vertex)
                case 4: return self.floyd_warshall_min_dist_algorithm(start_vertex)
                case 5: return self.a_star_min_dist_algorithm(start_vertex, goal_vertex)
                case _: logger.error("Algorithm doesn't exist returning no solution"); return None

        if self.graph.direction and self.graph.acyclical:
            logger.info("The graph is directed and acyclic, using Topological Sort to find minimum distance")
            return self.topological_sort_min_dist_algorithm(start_vertex)
        if self.graph.direction and not self.graph.has_negative_weight:
            logger.info("The graph is directed and doesn't have negative weights, using Dijkstra's to find minimum distance")
            return self.dijkstras_min_dist_algorithm(start_vertex)
        if self.graph.has_negative_weight:
            logger.info("The graph is either undirected or there is negative weighted edges, using Bellman-Ford's to find minimum distance")
            return self.bellman_fords_min_dist_algorithm(start_vertex)
        
        logger.error("ERROR: Automatic selection didn't run any minimum distance algorithm")
        return None

    def topological_sort_min_dist_algorithm(self, start_vertex:int) -> dict:
        distances = {v.id: float('inf') for v in self.graph.vertices.values()}
        distances[start_vertex] = 0
        for vertex_id in self.graph.topological_sort():
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
            distance_matrix[v1_id, v2_id] = e.weight
            previous_vertex[v1_id, v2_id] = v1_id
        for id, v in self.graph.vertices.items():
            distance_matrix[id, id] = 0
            previous_vertex[id, id] = id
        
        for k in range(v_size):
            for i in range(v_size):
                for j in range(v_size):
                    if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
                        previous_vertex[i, j] = previous_vertex[k, j]
        
        return {id: min_dist for id, min_dist in zip(self.graph.vertices.keys() ,distance_matrix[start_vertex, :])}


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
        T = Tree(Id="MST", V=[v_0], E=[])

        while not T.vertices.issuperset(self.graph.vertices):
            delta_edges = delta(T.vertices, self)
            # add the edge e that has the minimum weight in delta(V_T)
            # T = T + e
        
        return T


    def kruskals_min_spanning_tree_algorithm(self):
        pass


class NetworkFlowAlgorithms:
    def __init__(self, G: Graph):
        self.graph = G