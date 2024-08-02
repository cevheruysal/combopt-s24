import logging 
import random as r
from typing import Dict, List, Optional, Set, Tuple, Union

from config import RANDOM_SEED
from notation import Arc, Network, Vertex, Edge, Graph
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


def delta(Edges:Union[List[Edge], List[Arc]], 
          V_from:Union[List[int], Set[int]], 
          V_to:  Union[List[int], Set[int]]) -> Dict[str, Union[Set[Edge], Set[Arc]]]:
    """
    notation for the edges that are incident to a given vertices list V_from and V_to 
    in a comprehensive graph G that presumably contains all edges in the system """

    delta_edges = {"in":set(), "out":set(), "un-bi":set()}
    V_from_set, V_to_set = set(V_from), set(V_to)
    V_from_diff, V_to_diff = V_from_set.difference(V_to_set), V_to_set.difference(V_from_set)

    for e in Edges:
        (v1, v2) = e.incident_vertex_ids
        v1_contained_from = v1 in V_from_diff
        v2_contained_from = v2 in V_from_diff

        v1_contained_to = v1 in V_to_diff
        v2_contained_to = v2 in V_to_diff
        
        if v1_contained_from and not v2_contained_from:
            if v2_contained_to: 
                if e.direction is EdgeDirection.DIRECTED:
                    delta_edges["out"].add(e.copy())
                else: delta_edges["un-bi"].add(e.copy())
        if not v1_contained_from and v2_contained_from:
            if v1_contained_to: 
                if e.direction is EdgeDirection.DIRECTED:
                    delta_edges["in"].add(e.copy())
                else: delta_edges["un-bi"].add(e.copy())

    return delta_edges


def cut(N:Network, C:List[Vertex]) -> Tuple[Set[Arc], int, bool]:
    N_vertices_set = set(N.vertices.keys())
    C_set = {c.id for c in C}
    C_bar = N_vertices_set.difference(C_set)
    
    delta = delta(N.edges, C_set, C_bar)

    cut_edges = delta["out"]
    cut_capacity = sum([a.capacity for a in cut_edges])
    s_t_cut = N.source_node_id in C_set and N.sink_node_id in C_bar
    
    return cut_edges, cut_capacity, s_t_cut, 


def minimum_cost_edge_in_delta(delta:Dict[str, Set[Edge]]) -> Optional[Edge]:
    min_edge = None
    min_weight = float("inf")

    for set in delta.values():
        for edge in set:
            if edge.weight < min_weight:
                min_edge = edge
                min_weight = min_edge.weight

    return min_edge.copy()