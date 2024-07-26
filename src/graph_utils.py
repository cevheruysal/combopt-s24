import logging 
import random as r
from typing import Dict, List, Optional, Set, Tuple

from config import RANDOM_SEED
from notation import Vertex, Edge, Graph
from enums import EdgeDirection


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_random_directed_connected_graph(Id, num_vertices:int, num_edges:int, weight_range:Tuple[float, float] = (0, 1)):
    assert num_edges >= num_vertices - 1, "Number of edges must be at least num_vertices - 1 to ensure connectivity"
    r.seed(RANDOM_SEED)

    V = [Vertex(i) for i in range(num_vertices)]
    E = []
    added_edges = set()

    # Create a spanning tree to ensure connectivity
    for i in range(1, num_vertices):
        v1 = r.randint(0, i - 1)
        v2 = i
        weight = r.uniform(*weight_range)
        edge = Edge(Id=len(E), V1=v1, V2=v2, W=weight, D=EdgeDirection.DIRECTED)
        E.append(edge)
        added_edges.add((v1, v2))

    # Add remaining edges randomly
    while len(E) < num_edges:
        v1, v2 = r.sample(range(num_vertices), 2)
        if (v1, v2) not in added_edges and (v2, v1) not in added_edges:
            weight = r.uniform(*weight_range)
            edge = Edge(Id=len(E), V1=v1, V2=v2, W=weight, D=EdgeDirection.DIRECTED)
            E.append(edge)
            added_edges.add((v1, v2))

    return Graph(Id, V, E)

def delta(V_prime:Dict[int,Vertex], G:Graph) -> Dict[str, Set[Edge]]:
    delta_edges = {"in":set(), "out":set(), "un-bi":set()}
    
    for e in G.edges.values():
        (v1, v2) = e.end_vertex_ids
        v1_contained = v1 in V_prime.keys()
        v2_contained = v2 in V_prime.keys()
        
        if v1_contained:
            if v2_contained: continue
            elif e.direction is EdgeDirection.DIRECTED:
                delta_edges["out"].add(e.copy())
            else: delta_edges["un-bi"].add(e.copy())
        elif v2_contained:
            if e.direction is EdgeDirection.DIRECTED:
                delta_edges["in"].add(e.copy())
            else: delta_edges["un-bi"].add(e.copy())

    return delta_edges

def minimum_cost_edge_in_delta(delta:Dict[str, Set[Edge]]) -> Optional[Edge]:
    min_edge = None
    min_weight = float("inf")

    for set in delta.values():
        for edge in set:
            if edge.weight < min_weight:
                min_edge = edge
                min_weight = min_edge.weight

    return min_edge.copy()