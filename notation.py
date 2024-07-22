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
        return self.end_vertex_ids[0] * C + self.end_vertex_ids[1]
        
    def copy(self):
        V1, V2 = self.end_vertex_ids
        Id, W, D = self.id, self.weight, self.direction
        return Edge(Id, V1, V2, W, D)
    
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
        self.topological_order = None
        self.acyclical = None

        self.init_vertices(V)
        self.init_edges(E)
        self.update_meta()

    def init_vertices(self, V:List[Vertex]) -> None:
        self.vertices = {v.id: v for v in sorted(V, key=lambda x: x.id)}
    
    def init_edge(self, e:Edge):
        v1, v2 = e.end_vertex_ids
        self.vertices[v1].leafs.add(v2)
        self.vertices[v2].roots.add(v1)
        
        if e.direction != EdgeDirection.DIRECTED: #?????????
            self.vertices[v1].roots.add(v2)
            self.vertices[v2].leafs.add(v1)

        self.edges[((e.end_vertex_ids[0], e.end_vertex_ids[1]))] = e

    def init_edges(self, E:List[Edge]) -> None:
        constant = max(self.vertices.keys(), default=0)
        E.sort(key=lambda e: e.sort_key(constant))
        for e in E: self.init_edge(e)
        
    def get_graph_direction(self) -> GraphDirection:
        first_value = None
        for edge in self.edges.values():
            if first_value is None:
                first_value = edge.direction
            elif edge.direction != first_value:
                return GraphDirection.MIXED
        return GraphDirection[first_value.name]
    
    def isCyclicUtil(self, dir:int, v:Vertex, visited:Dict[int, bool], 
                     recStack:Dict[int, bool], parent:int):
        visited[v.id] = True

        for leaf_id in v.leafs:
            leaf_vertex = self.vertices[leaf_id]
            if dir == 1:
                recStack[v.id] = True
                
                if visited[leaf_id] is False:
                    if self.isCyclicUtil(leaf_vertex, visited, recStack) is True:
                        return True    
                
                elif recStack[leaf_id] is True:
                    return True
                
                recStack[v] = False

            elif dir == 2:
                if visited[leaf_id] is False:
                    if self.isCyclicUtil(leaf_vertex, visited, parent=v.id) is True:
                        return True
                
                elif parent != leaf_id:
                    return True
            else: 
                return False

        return False

    def isCyclic(self) -> bool:
        return self.topological_order is not None

        visited = {id: False for id in self.vertices.keys()}
        recStack = {id: False for id in self.vertices.keys()}
        directed = (1 if self.direction == GraphDirection.DIRECTED 
                    else 2 if self.direction in (GraphDirection.UNDIRECTED, GraphDirection.BIDIRECTED) 
                    else 0)
        for vertex in self.vertices.values():
            if visited[vertex.id] is False:
                if self.isCyclicUtil(directed, vertex, visited, recStack, parent=-1) is True:
                    return True
        return False
    
    def topological_sort(self) -> Optional[List[int]]:
        if len(self.edges) == 0:
            logger.warning("No edges present to perform topological sorting")
            return None
        
        sorted_vertices = []
        all_vertices_dict = {id: v.copy() for id, v in self.vertices.items()}
        no_root_vertices_list = [v.copy() for v in self.vertices.values() if len(v.roots) == 0]
        remaining_edges_dict = {(e.end_vertex_ids[0], e.end_vertex_ids[1]): 1 for e in self.edges.values()}

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
        Graph.__init__(self, Id, V, E)


class Tree(Forest):
    def __init__(self, Id: int, V: List[Vertex] = [], E: List[Edge] = []):
        # trees are cycle-free AND connected graphs
        Forest.__init__(self, Id, V, E)


"""
if __name__ == "__main__":
    v0 = Vertex(0)
    v1 = Vertex(1)
    v2 = Vertex(2)
    v3 = Vertex(3)
    v4 = Vertex(4)

    e0 = Edge(0, 0, 1, 1, EdgeDirection.DIRECTED)
    e1 = Edge(1, 0, 2, 0.5, EdgeDirection.DIRECTED)
    e2 = Edge(2, 1, 2, 2, EdgeDirection.UNDIRECTED)
    e3 = Edge(3, 2, 3, 1.5, EdgeDirection.BIDIRECTED)
    e4 = Edge(4, 3, 4, -1, EdgeDirection.DIRECTED)

    V = [v0, v1, v2, v3, v4]
    E = [e0, e1, e2, e3, e4]

    G = Graph(0, V, E)
    G.disp()

    #Exercise 1.3
    v_S = Vertex(0)
    v_1 = Vertex(1)
    v_2 = Vertex(2)
    v_3 = Vertex(3)
    v_4 = Vertex(4)
    v_5 = Vertex(5)
    v_6 = Vertex(6)
    v_7 = Vertex(7)
    v_8 = Vertex(8)
    v_9 = Vertex(9)
    v_T = Vertex(10)

    e_0 = Edge(0, 0, 1, 2)
    e_1 = Edge(1, 0, 2, 4)
    e_2 = Edge(2, 0, 3, 6)
    e_3 = Edge(3, 1, 2, 2)
    e_4 = Edge(4, 1, 4, 4)
    e_5 = Edge(5, 2, 4, 3)
    e_6 = Edge(6, 2, 5, 3)
    e_7 = Edge(7, 3, 4, 7)
    e_8 = Edge(8, 3, 6, 3)
    e_9 = Edge(9, 4, 5, 1)
    e_10 = Edge(10, 4, 7, 4)
    e_11 = Edge(11, 5, 3, -3)
    e_12 = Edge(12, 5, 8, 1)
    e_13 = Edge(13, 6, 7, 1)
    e_14 = Edge(14, 6, 9, 1)
    e_15 = Edge(15, 7, 5, 4)
    e_16 = Edge(16, 7, 10, 3)
    e_17 = Edge(17, 8, 6, 1)
    e_18 = Edge(18, 8, 9, 6)
    e_19 = Edge(19, 8, 10, 6)
    e_20 = Edge(20, 9, 10, 4)


    V_exercise1 = [v_S, v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8, v_9, v_T]
    E_exercise1 = [e_0, e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9, e_10, e_11, e_12, e_13, e_14, e_15, e_16, e_17, e_18, e_19, e_20]
    G_exercise1 = Graph(1, V_exercise1, E_exercise1)

    G_exercise1.disp()
        
    G2 = Graph(2)
    G2.build_random_digraph(1, RandomRange(1, 10, 0.5))
    G2.disp()
    """
