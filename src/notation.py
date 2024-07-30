import logging
from typing import Optional, List, Dict, Set, Tuple, Union

from enums import EdgeDirection, GraphDirection
from config import ROUND_TO
from util_structs import UnionFind


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
        return hash((self.id, tuple(self.roots), tuple(self.leafs)))
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
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
        self.id = Id
        self.incident_vertex_ids = (V1, V2)
        self.weight = W
        self.direction = D
    
    def sort_key(self, C) -> int:
        return self.weight * C**2 + self.incident_vertex_ids[0] * C + self.incident_vertex_ids[1]
        
    def copy(self):
        V1, V2 = self.incident_vertex_ids
        Id, W, D = self.id, self.weight, self.direction
        return Edge(Id, V1, V2, W, D)
    
    def __hash__(self):
        return hash(self)
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.id == other.id and 
                    self.incident_vertex_ids == other.incident_vertex_ids and 
                    self.weight == other.weight and 
                    self.direction == other.direction)
        return False
    
    def __str__(self):
        V1, V2 = self.incident_vertex_ids
        Id, W, D = self.id, round(float(self.weight), ROUND_TO), self.direction
        sign = "+" if W > 0 else ""
        d1, d2 = "<" if D.value == 2 else "-", "-" if D.value == 0 else ">" 
        return f"E{Id}:= V{V1} {d1}---W:{sign}{W}---{d2} V{V2}\n"
    

class Arc(Edge):
    def __init__(self, Id: int, V1: int, V2: int, C:float, W: float = 0):
        super().__init__(Id, V1, V2, W)
        self.capacity = C


class Graph:
    def __init__(self, Id:Union[int,str], V:List[Vertex] = [], E:List[Edge] = []):
        self.id: int = Id 
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.connected_components: UnionFind = None
        self.has_negative_weight: Optional[bool] = None
        self.direction: Optional[str] = None
        self.acyclical: Optional[bool] = None

        self.init_vertices(V)
        self.init_edges(E)
        self.update_meta()

    def init_vertices(self, V:List[Vertex]) -> None:
        for v in sorted(V, key=lambda x: x.id):
            self.vertices[v.id] = v
        self.connected_components = UnionFind(self.vertices.keys())
    
    def add_edge(self, e:Edge):
        try:
            v1, v2 = e.incident_vertex_ids

            for v in [v1, v2]:
                if v not in self.vertices:
                    logger.error(f"Vertex not found: {v}")
                    self.vertices[v] = Vertex(v)
                    logger.warning(f"New vertex created with id: {v}")
                    self.connected_components.add(v)

            self.connected_components.union(v1, v2)
            self.vertices[v1].leafs.add(v2)
            self.vertices[v2].roots.add(v1)

            if e.direction != EdgeDirection.DIRECTED:
                self.vertices[v1].roots.add(v2)
                self.vertices[v2].leafs.add(v1)

            self.edges[(v1, v2)] = e

        except Exception as ex:
            logger.error(f"An error occurred while initializing edge: {ex}")
            raise

    def init_edges(self, E:List[Edge]) -> None:
        constant = max(self.vertices.keys(), default=0)
        E.sort(key=lambda e: e.sort_key(constant))
        for e in E: 
            self.add_edge(e)
        
    def get_graph_direction(self) -> GraphDirection:
        directions = {edge.direction for edge in self.edges.values()}
        if len(directions) == 1:
            return GraphDirection[directions.pop().name]
        return GraphDirection.MIXED

    def is_cyclic(self) -> bool:
        """
        A digraph is acyclic if it does not contain a (directed) cycle
        similarly an undirected graph is acyclic if doesn't contain a cycle """

        visited, rec_stack, parent = set(), set(), {}

        def is_cyclic_dfs(v:Vertex) -> bool:
            visited.add(v.id); rec_stack.add(v.id)
            
            for leaf_id in v.leafs:
                leaf_vertex = self.vertices[leaf_id]
                if (v.id, leaf_id) in self.edges: 
                    edge_type = self.edges[(v.id, leaf_id)].direction
                else: edge_type = self.edges[(leaf_id, v.id)].direction

                if edge_type == EdgeDirection.DIRECTED:
                    if leaf_id not in visited:
                        if is_cyclic_dfs(leaf_vertex): return True
                    elif leaf_id in rec_stack:
                        return True
                else: # elif edge_type in {EdgeDirection.UNDIRECTED, EdgeDirection.BIDIRECTED}:
                    if leaf_id not in visited:
                        parent[leaf_id] = v.id
                        if is_cyclic_dfs(leaf_vertex): return True
                    elif parent.get(v.id) != leaf_id:
                        return True

            rec_stack.remove(v.id)
            return False

        for v in self.vertices.values():
            if v.id not in visited:
                if is_cyclic_dfs(v):
                    return True
        return False
    
    def get_connected_components(self):
        unique_components = {self.connected_components.find(v_id) for v_id in self.vertices}
        return len(unique_components)

    def update_meta(self):
        self.has_negative_weight = any(e.weight < 0 for e in self.edges.values())
        self.direction = self.get_graph_direction()
        self.acyclical = not self.is_cyclic()
        self.connected = self.get_connected_components() <= 1

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
        """
        Let 𝐺 = (𝑉, 𝐸) be an undirected graph.
        (i) 𝐺 is called forest if it is cycle-free.
        (ii) A subgraph 𝐹 of 𝐺 with 𝑉(𝐹) = 𝑉 is called spanning forest of 𝐺 if 𝐹 is a forest """

        super().__init__(Id, V, E)
        if not self.acyclical:
            raise ValueError("A forest cannot contain cycles")


class Tree(Forest):
    def __init__(self, Id: int, V: List[Vertex] = [], E: List[Edge] = []):
        """
        Let 𝐺 = (𝑉, 𝐸) be an undirected graph, then the following are equivalent:
        (i) 𝐺 is a tree.
        (ii) 𝐺 is connected and cycle-free.
        (iii) 𝐺 is connected and removing any edge would result in a non-connected graph.
        (iv) 𝐺 is cycle-free and adding any edge would produce a cycle.
        (v) 𝐺 is connected and |𝐸| = |𝑉| − 1.
        (vi) 𝐺 is cycle-free and |𝐸| = |𝑉| − 1 """
        
        super().__init__(Id, V, E)
        if not self.connected:
            raise ValueError("A tree must be connected")


class Network():
    def __init__(self, Id, V, A, s, t, u):
        """
        A Network is a tuple 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) where 𝐺 = (𝑉, 𝐸) is a digraph, 
        𝑠, 𝑡 ∈ 𝑉 with 𝑠 ≠ 𝑡 are two distinct nodes in 𝐺 (called source and terminal or sink, respectively), 
        and 𝑢∶ 𝐸 → Q≥0 is called arc capacities.

        Furthermore, an 𝑠-𝑡-flow in the network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) is a function 𝑓∶ 𝐸 → Q≥0 
        with the following properties:
        1. 0 ≤ 𝑓(𝑒) ≤ 𝑢(𝑒) for each arc 𝑒 ∈ 𝐸 (capacity constraints)
        2. 𝑓 (𝛿_out(𝑣)) = 𝑓 (𝛿in(𝑣)) for each node 𝑣 ∈ 𝑉 ∖ {𝑠, 𝑡} (flow conservation constraints)
            We call 𝑓 (𝛿_out(𝑠)) − 𝑓 (𝛿_in(𝑠)) the value of the flow 𝑓, 
            that is the net amount of flow that leaves the source node 𝑠. """
        
        """
        Let 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) be a network and let 𝑓 be an 𝑠-𝑡-flow in 𝑁.
        (i) For an arc 𝑒 = (𝑣, 𝑤) ∈ 𝐸(𝐺) we denote the reverse arc by ⃖𝑒 ∶= (𝑤, 𝑣).
        (ii) The residual network 𝑁_𝑓 = (𝐺_𝑓, 𝑠, 𝑡, 𝑢_𝑓) is the network with
            1 𝑉(𝐺_𝑓) = 𝑉(𝐺)
            2 𝐸(𝐺_𝑓) = {𝑒 ∈ 𝐸(𝐺) ∶ 𝑓(𝑒) < 𝑢(𝑒)} ∪ {⃖𝑒 ∶ 𝑒 ∈ 𝐸(𝐺) ∧ 𝑓(𝑒) > 0}
            3 𝑢_𝑓(𝑣, 𝑤) =  {𝑢(𝑣, 𝑤) − 𝑓(𝑣, 𝑤)  for all (𝑣, 𝑤) ∈ 𝐸(𝐺)  with 𝑓(𝑣, 𝑤) < 𝑢(𝑣, 𝑤)
                            {𝑓(𝑤, 𝑣)            for all (𝑤, 𝑣) ∈ 𝐸(𝐺)  with 𝑓(𝑤, 𝑣) > 0
        (iii) An 𝑓-augmenting path is an 𝑠-𝑡-path in the residual network 𝑁_𝑓."""

        """
        Let 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) be a network, 
        𝑓 ∶ 𝐸(𝐺) → 𝑄≥0 a feasible flow in 𝑁 and 
        let 𝑃 be an 𝑓-augmenting path. 
        
        Then the flow ̃𝑓 ∶ 𝐸(𝐺) → Q≥0 obtained from 𝑓 by augmenting along 𝑃 is given by
        ̃𝑓 (𝑒) := ⎧𝑓 (𝑒) + 𝛾, if 𝑒 ∈ 𝐸(𝑃),
                 ⎨𝑓 (𝑒) − 𝛾, if ⃖𝑒 ∈ 𝐸(𝑃),
                 ⎩𝑓 (𝑒),     otherwise;
        
        where 𝛾 ∶= min{𝑢_𝑓(𝑒) ∶ 𝑒 ∈ 𝐸(𝑃)} """

        # initiate like a graph 
        # but its a network deal with antiparallel edges as well in the residual network
        self.residual_network = Graph()
        pass
