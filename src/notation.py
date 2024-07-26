import logging
from typing import Optional, List, Dict, Set, Tuple, Union
import random as r

from enums import EdgeDirection, GraphDirection
from config import EDGE_PROB, EDGE_ORIENTATION_PROB, RANDOM_SEED, ROUND_TO

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Vertex:
    def __init__(self, Id: int, RootNeighbors: Optional[Set[int]] = None, LeafNeighbors: Optional[Set[int]] = None):
        self.id = Id
        self.roots = RootNeighbors if RootNeighbors is not None else set()
        self.leafs = LeafNeighbors if LeafNeighbors is not None else set()
    
    def copy(self):
        return Vertex(self.id, self.roots.copy(), self.leafs.copy())
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Vertex):
            return (self.id == other.id and 
                    self.roots == other.roots and 
                    self.leafs == other.leafs)
        return False
    
    def __str__(self):
        roots_line = "Roots: " + ", ".join(["v" + str(v) for v in self.roots]) if len(self.roots) > 0 else "Roots: None"
        leafs_line = "Leafs: " + ", ".join(["v" + str(v) for v in self.leafs]) if len(self.leafs) > 0 else "Leafs: None"
        space_count = (len(roots_line) + len(leafs_line)) // 4 - 1
        vertx_line = "Id: " + " " * space_count + "V" + str(self.id)

        return roots_line + "\n" + vertx_line + "\n" + leafs_line + "\n"


class Edge:
    def __init__(self, Id: int, V1: int, V2: int, W: float, D: EdgeDirection = EdgeDirection.DIRECTED):
        self.end_vertex_ids = (V1, V2)
        self.weight = W
        self.direction = D
        self.id = Id
    
    def sort_key(self, C) -> int:
        return self.weight * C**2 + self.end_vertex_ids[0] * C + self.end_vertex_ids[1]
        
    def copy(self):
        V1, V2 = self.end_vertex_ids
        Id, W, D = self.id, self.weight, self.direction
        return Edge(Id, V1, V2, W, D)
    
    def __hash__(self):
        return hash(self)
    
    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.id == other.id and 
                    self.end_vertex_ids == other.end_vertex_ids and 
                    self.weight == other.weight and 
                    self.direction == other.direction)
        return False
    
    def __str__(self):
        V1, V2 = self.end_vertex_ids
        Id, W, D = self.id, round(float(self.weight), ROUND_TO), self.direction
        sign = "+" if W > 0 else ""
        d1, d2 = "<" if D.value == 2 else "-", "-" if D.value == 0 else ">" 
        return f"E{Id}:= V{V1} {d1}---W:{sign}{W}---{d2} V{V2}\n"


class Graph:
    def __init__(self, Id:Union[int,str], V:List[Vertex] = [], E:List[Edge] = []):
        self.id = Id 
        self.vertices = {} 
        self.edges = {}
        self.has_negative_weight = None
        self.direction = None
        self.acyclical = None

        self.init_vertices(V)
        self.init_edges(E)
        self.update_meta()

    def init_vertices(self, V:List[Vertex]) -> None:
        self.vertices = {v.id: v for v in sorted(V, key=lambda x: x.id)}
    
    def init_edge(self, e:Edge):
        try:
            v1, v2 = e.end_vertex_ids

            for v in [v1, v2]:
                if v not in self.vertices:
                    logger.error(f"Vertex not found: {v}")
                    self.vertices[v] = Vertex(v)
                    logger.warning(f"New vertice is created with id: {v}")

            self.vertices[v1].leafs.add(v2)
            self.vertices[v2].roots.add(v1)
            
            if e.direction != EdgeDirection.DIRECTED:
                self.vertices[v1].roots.add(v2)
                self.vertices[v2].leafs.add(v1)
            
            self.edges[(v1, v2)] = e

        except Exception as ex:
            logger.error(f"An error occurred while initializing edge: {ex}")

    def init_edges(self, E:List[Edge]) -> None:
        constant = max(self.vertices.keys(), default=0)
        E.sort(key=lambda e: e.sort_key(constant))
        for e in E: 
            self.init_edge(e)
        
    def get_graph_direction(self) -> GraphDirection:
        directions = {edge.direction for edge in self.edges.values()}
        if len(directions) == 1:
            return GraphDirection[directions.pop().name]
        return GraphDirection.MIXED

    def isCyclic(self) -> bool:
        visited, rec_stack, parent = set(), set(), {}

        def isCyclicDFS(v:Vertex):
            visited.add(v.id); rec_stack.add(v.id)
            
            for leaf_id in v.leafs:
                leaf_vertex = self.vertices[leaf_id]
                if (v.id, leaf_id) in self.edges: 
                    edge_type = self.edges[(v.id, leaf_id)].direction
                else: edge_type = self.edges[(leaf_id, v.id)].direction

                if edge_type == EdgeDirection.DIRECTED:
                    if leaf_id not in visited:
                        if isCyclicDFS(leaf_vertex): return True
                    elif leaf_id in rec_stack: return True
                    
                elif (edge_type == EdgeDirection.UNDIRECTED or 
                      edge_type == EdgeDirection.BIDIRECTED):
                    if leaf_id not in visited:
                        parent[leaf_id] = v.id
                        if isCyclicDFS(leaf_vertex): return True
                    elif parent[v.id] != leaf_id: return True

            rec_stack.remove(v.id)
            return False

        for v in self.vertices.values():
            if v.id not in visited:
                if isCyclicDFS(v):
                    return True
        return False

    def update_meta(self):
        self.has_negative_weight = any(e.weight < 0 for e in self.edges.values())
        self.direction = self.get_graph_direction()
        self.acyclical = self.isCyclic()

    def copy(self):
        return Graph(self.id, [v.copy() for v in self.vertices.values()], 
                     [e.copy() for e in self.edges.values()])
    
    def __str__(self):
        graph_info = (f"Metadata of Graph{self.id}:\n"
                      + ("This is a directed " if self.direction else "This is an undirected ")
                      + ("acyclic graph " if self.acyclical else "graph that possibly includes cycles ")
                      + "with " + ("some negative " if self.has_negative_weight else "all positive ") + "edge weights\n"
                      )
        return graph_info

    def disp(self, short_disp=True):
        print("-"*100)
        print(self.__str__())
        if not short_disp:
            for v in self.vertices.values():
                print(v)
        for e in self.edges.values():
            print(e)


class Forest(Graph):
    def __init__(self, Id: int, V: List[Vertex] = [], E: List[Edge] = []):
        # forests are cycle-free graphs
        super().__init__(Id, V, E)
        if self.isCyclic():
            raise ValueError("A forest cannot contain cycles")


class Tree(Forest):
    def __init__(self, Id: int, V: List[Vertex] = [], E: List[Edge] = []):
        # trees are cycle-free AND connected graphs
        super().__init__(Id, V, E)
        if not self.is_connected():
            raise ValueError("A tree must be connected")
