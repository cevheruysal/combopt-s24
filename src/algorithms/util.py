from collections import deque
from typing import Optional, List, Tuple, Dict
import logging

from src.notation import Graph, Network

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

    logger.info("Topological sorting finished, no possible sorting found. The graph may have cycles.")
    return None


@staticmethod
def find_st_path(N: Network) -> Optional[Tuple[List[Tuple[int, int]], float]]:
    def construct_path_to_node(parent: Dict[int, int], node: int) -> List[Tuple[int, int]]:
        path = []
        while parent[node] != -1:
            path.insert(0, (parent[node], node))
            node = parent[node]
        return path

    parent_dict = {v: -1 for v in N.vertices.keys()}
    visited = set()
    queue = deque([(N.source_node_id, float('inf'))])

    while queue:
        current, flow = queue.popleft()
        if current in visited: continue
        visited.add(current)

        for neighbor in N.vertices[current].leafs:
            arc = N.edges[(current, neighbor)]

            if neighbor not in visited and arc.remaining_capacity() > 0:
                new_flow = min(flow, arc.remaining_capacity())
                parent_dict[neighbor] = current

                if neighbor == N.sink_node_id:
                    path = construct_path_to_node(parent_dict, N.sink_node_id)
                    logger.info(f"Found augmenting path with flow {new_flow}")
                    return path, new_flow

                queue.append((neighbor, new_flow))
    return None, 0


@staticmethod
def blocking_flow(N: Network, node: int, flow: int, start: Dict[int, int]) -> int:
    if node == N.sink_node_id:
        return flow

    # for v in self.network.vertices[u].leafs:
    while start[node] < len(N.vertices[node].leafs):
        leaf = N.vertices[node].leafs[start[node]]

        if (N.node_levels[leaf] == N.node_levels[node] + 1 and
                N.edges[(node, leaf)].remaining_capacity() > 0):
            curr_flow = min(flow, N.edges[(node, leaf)].remaining_capacity())
            temp_flow = blocking_flow(N, leaf, curr_flow, start)

            if temp_flow > 0:
                N.edges[(node, leaf)].alter_flow(-temp_flow)
                N.edges[(leaf, node)].alter_flow(+temp_flow)
                return temp_flow

        start[node] += 1
    return 0