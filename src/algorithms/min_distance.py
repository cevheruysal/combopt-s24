import heapq
import logging
import numpy as np
import util

from typing import Optional, List
from src.enums import MinDistanceAlgorithmsEnum
from src.notation import Graph

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MinDistanceAlgorithmSolver:
    def __init__(self, G: Graph):
        """
        Input: an undirected, connected graph 𝐺 = (𝑉, 𝐸) with edge weights 𝑐 ∶ 𝐸 → Q.
        Task: Find a spanning tree 𝑇 in 𝐺 such that 𝑐(𝐸(𝑇)) = ∑_{𝑒∈𝐸(𝑇)} 𝑐(𝑒) is minimized """
        self.algorithms = [topological_sort, dijkstra, bellman_fords, floyd_warshall, a_star_min_dist_algorithm]
        self.graph = G

    def auto_detect(self) -> MinDistanceAlgorithmsEnum:
        algo = None
        if self.graph.direction and self.graph.acyclical:
            logger.info("The graph is directed and acyclic, using Topological Sort to find minimum distance")
            algo = MinDistanceAlgorithmsEnum.TOPOLOGICAL_SORT
        elif self.graph.direction and not self.graph.has_negative_weight:
            logger.info(
                "The graph is directed and doesn't have negative weights, using Dijkstra's to find minimum distance")
            algo = MinDistanceAlgorithmsEnum.DIJKSTRA
        elif self.graph.has_negative_weight:
            logger.info(
                "The graph is either undirected or there is negative weighted edges, using Bellman-Ford's to find "
                "minimum distance")
            algo = MinDistanceAlgorithmsEnum.BELLMAN_FORD
        else:
            logger.error("Automatic selection didn't run any minimum distance algorithm")
        return algo

    def run(self, start_vertex: int, goal_vertex: int,
            use_algorithm: MinDistanceAlgorithmsEnum = MinDistanceAlgorithmsEnum.AUTOMATIC) -> Optional[dict]:
        result = None

        if use_algorithm.value < 1:
            logger.info(f"Trying to automatically determine the algorithm to find minimum distance")
            use_algorithm = self.auto_detect()
        logger.info(f"Trying to use {use_algorithm.name} to find minimum distance")

        # TODO: Calling convention for varying argument lengths
        return self.algorithms[use_algorithm.value - 1](self.graph, start_vertex, goal_vertex)


def topological_sort(G, start_vertex: int) -> dict:
    """
    Let 𝐺 be an acyclic directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q and let 𝑠, 𝑡 ∈ 𝑉(𝐺).
    Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑛 + 𝑚) """

    distances = {v.id: float('inf') for v in G.vertices.values()}
    distances[start_vertex] = 0

    for vertex_id in util.topological_sort(G):
        if distances[vertex_id] == float('inf'):
            continue

        for neighbor in G.vertices[vertex_id].leafs:
            edge = G.edges[(vertex_id, neighbor)]
            new_distance = distances[vertex_id] + edge.weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance

    logger.info("Found solution using Topological Sort")
    return distances


def dijkstra(G, start_vertex: int) -> dict:
    """
    Let 𝐺 be a directed graph with edge weights 𝑐 ∶ 𝐸(𝐺) → Q≥0 and let 𝑠, 𝑡 ∈ 𝑉(𝐺).
    Then we can compute a shortest 𝑠-𝑡-path in 𝐺 in time 𝒪 (𝑚 + 𝑛 log 𝑛) """

    distances = {v.id: float('inf') for v in G.vertices.values()}
    distances[start_vertex] = 0
    heap = [(0, start_vertex)]
    heapq.heapify(heap)

    while heap:
        current_distance, current_vertex = heapq.heappop(heap)
        if current_distance > distances[current_vertex]:
            continue

        for neighbor in G.vertices[current_vertex].leafs:
            edge = G.edges[(current_vertex, neighbor)]
            distance = current_distance + edge.weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))

    logger.info("Found solution using Dijkstra's")
    return distances


def bellman_fords(G, start_vertex: int) -> Optional[dict]:
    """
    Let 𝐺 = (𝑉, 𝐸) be a directed graph with edge weights 𝑐 ∶ 𝐸 → Q and let 𝑠, 𝑡 ∈ 𝑉.
    There is an algorithm that either computes a shortest 𝑠-𝑡-path in 𝐺
    or finds a negative cycle in 𝐺 in time 𝒪 (𝑚𝑛) """

    distances = {v.id: float('inf') for v in G.vertices.values()}
    distances[start_vertex] = 0

    for _ in range(len(G.vertices) - 1):
        for edge in G.edges.values():
            u, v = edge.incident_vertex_ids
            distances[v] = min(distances[u] + edge.weight,
                               distances[v])

    # Check for negative weight cycles
    for edge in G.edges.values():
        u, v = edge.incident_vertex_ids
        if distances[u] + edge.weight < distances[v]:
            logger.info("Graph contains a negative weight cycle")
            return None

    logger.info("Found solution using Bellman Ford's")
    return distances


def floyd_warshall(G, start_vertex: int) -> Optional[dict]:
    v_size = len(G.vertices)
    distance_matrix = np.ones([v_size, v_size]) * np.inf
    previous_vertex = np.array([-1] * (v_size ** 2)).reshape(v_size, v_size)

    for (v1_id, v2_id), e in G.edges.items():
        distance_matrix[v1_id - 1, v2_id - 1] = e.weight
        previous_vertex[v1_id - 1, v2_id - 1] = v1_id
    for id, v in G.vertices.items():
        distance_matrix[id - 1, id - 1] = 0
        previous_vertex[id - 1, id - 1] = id

    for k in range(v_size):
        for i in range(v_size):
            for j in range(v_size):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
                    previous_vertex[i, j] = previous_vertex[k, j]

    return {id: min_dist for id, min_dist in zip(G.vertices.keys(), distance_matrix[start_vertex - 1, :])}


def a_star_min_dist_algorithm(G, start_vertex: int, goal_vertex: int, heuristic) -> Optional[List[int]]:
    open_set = []
    heapq.heappush(open_set, (0, start_vertex))
    came_from = {start_vertex: None}
    g_score = {v.id: float('inf') for v in G.vertices.values()}
    g_score[start_vertex] = 0
    f_score = {v.id: float('inf') for v in G.vertices.values()}
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

        for neighbor in G.vertices[current].leafs:
            edge = G.edges[(current, neighbor)]
            tentative_g_score = g_score[current] + edge.weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_vertex)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None
