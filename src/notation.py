from collections import deque
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
    def __init__(self, Id:int, 
                 RootNeighbors:Optional[Set[int]] = None, 
                 LeafNeighbors:Optional[Set[int]] = None):
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
    def __init__(self, Id:int,
                 V1:int, V2:int, W:float = 0, 
                 D:EdgeDirection = EdgeDirection.DIRECTED):
        self.id = Id
        self.incident_vertex_ids = (V1, V2)
        self.weight = W
        self.direction = D
    
    def sort_key(self, C:int) -> float:
        return self.weight * C**2 + \
               self.incident_vertex_ids[0] * C + \
               self.incident_vertex_ids[1]
        
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
    def __init__(self, Id:int, 
                 V1:int, V2:int, 
                 U:float = 0, F:float = 0, R:bool = False):
        super().__init__(Id, V1, V2)
        self.capacity = U
        self.flow = F
        self.residual_arc = R

    def sort_key(self, C:int) -> float:
        return self.capacity * C**2 + \
               self.incident_vertex_ids[0] * C + \
               self.incident_vertex_ids[1]
    
    def set_flow(self, value:float) -> None:
        if value < 0:
            logger.error("Flow value cannot be nagative")
            raise ValueError()
        
        if self.residual_arc:
            self.capacity = value
        else:
            if value > self.capacity: 
                logger.error("Flow value cannot surpass arc capacity")
                raise ValueError()
            self.flow = value

    def alter_flow(self, delta:float) -> None:
        if self.residual_arc:
            if self.capacity - delta < 0:
                logger.error("Negative residual arc is not possible")
                raise ValueError()
            
            self.capacity -= delta
        
        else:
            if self.flow + delta < 0: 
                logger.error("Negative flow is not possible on arc")
                raise ValueError()
            elif self.flow + delta < 0:
                logger.error("Flow cannot surpass capacity")
                raise ValueError()
        
            self.flow += delta
    
    def remaining_capacity(self) -> float:
        return self.capacity - self.flow

    def copy(self):
        V1, V2 = self.incident_vertex_ids
        Id, U, F, R  = self.id, self.capacity, self.flow, self.residual_arc
        return Arc(Id, V1, V2, U, F, R)
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.id == other.id and 
                    self.incident_vertex_ids == other.incident_vertex_ids and 
                    self.flow == other.flow and 
                    self.capacity == other.capacity and 
                    self.residual_arc == other.residual_arc)
        return False
    
    def __str__(self):
        if self.residual_arc and self.capacity <= 0:
            return ""
        V1, V2 = self.incident_vertex_ids
        Id, U, F = (self.id, round(float(self.capacity), ROUND_TO), 
                    round(float(self.flow), ROUND_TO))
        return f"A{Id}:= V{V1} ----F/U:{F}/{U}---> V{V2}\n"


class Graph:
    def __init__(self, Id:Union[int,str], 
                 V:List[Vertex] = [], E:List[Edge] = []):
        self.id: int = Id 
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self.connected_components = UnionFind([])
        self.has_negative_weight: Optional[bool] = None
        self.direction: Optional[str] = None
        self.acyclical: Optional[bool] = None

        self.init_vertices(V)
        self.init_edges(E)
        self.update_meta()

    def init_vertices(self, V:List[Vertex]) -> None:
        for v in sorted(V, key=lambda x: x.id):
            self.vertices[v.id] = v
            self.connected_components.add(v.id)
    
    def add_edge(self, e:Union[Edge, Arc]) -> None:
        v1, v2 = e.incident_vertex_ids

        for v in [v1, v2]:
            if v not in self.vertices:
                logger.warning(f"Vertex not found: {v}")
                self.vertices[v] = Vertex(v)
                logger.warning(f"New vertex created with id: {v}")
                self.connected_components.add(v)

        if not (isinstance(e, Arc) and e.residual_arc):
            self.connected_components.union(v1, v2)
            self.vertices[v1].leafs.add(v2)
            self.vertices[v2].roots.add(v1)

        if e.direction != EdgeDirection.DIRECTED:
            self.vertices[v1].roots.add(v2)
            self.vertices[v2].leafs.add(v1)

        self.edges[(v1, v2)] = e

    def init_edges(self, E:Union[List[Edge], List[Arc]]) -> None:
        constant = len(self.vertices)
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
    
    def get_connected_components(self) -> int:
        unique_components = {self.connected_components.find(v_id) for v_id in self.vertices}
        return len(unique_components)

    def update_meta(self) -> None:
        self.has_negative_weight = any(e.weight < 0 for e in self.edges.values())
        self.direction = self.get_graph_direction()
        self.acyclical = not self.is_cyclic()
        self.connected = self.get_connected_components() <= 1

    def copy(self):
        return Graph(self.id, 
                     [v.copy() for v in self.vertices.values()], 
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
    def __init__(self, Id:int, 
                 V:List[Vertex] = [], E:List[Edge] = []):
        """
        Let 𝐺 = (𝑉, 𝐸) be an undirected graph.
        (i) 𝐺 is called forest if it is cycle-free.
        (ii) A subgraph 𝐹 of 𝐺 with 𝑉(𝐹) = 𝑉 is called spanning forest of 𝐺 if 𝐹 is a forest """

        super().__init__(Id, V, E)
        if not self.acyclical:
            raise ValueError("A forest cannot contain cycles")


class Tree(Forest):
    def __init__(self, Id:int, 
                 V:List[Vertex] = [], E:List[Edge] = []):
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


class Network(Graph):
    def __init__(self, Id:Union[int,str], 
                 V:List[Vertex], A:List[Arc], 
                 s:Union[int, str], t:Union[int, str], f:int = 0):
        """
        !! 𝑢 argument is not used since kinda dumb to write it specifically !!
        
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

        self.id: int = Id 
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[Tuple[int, int], Arc] = {}
        self.connected_components = UnionFind([])
        self.node_levels: Dict[int, int] = None

        self.direction: Optional[str] = None
        self.acyclical: Optional[bool] = None
        self.st_connected: Optional[bool] = None
        
        self.init_vertices(V)
        self.init_arcs(A)

        if s not in self.vertices.keys() or t not in self.vertices.keys() or s == t:
            logger.error("Source and sink nodes must be specified for a proper Network")
            raise ValueError()
        
        self.source_node_id = s
        self.sink_node_id = t
        self.flow = f
        
        self.update_node_levels()
        self.update_meta()

        if not self.st_connected:
            logger.error("Source and sink nodes must be connected")
            raise ValueError()
        
    def refactor_antiparallel_arcs(self, A:List[Arc]) -> List[Arc]:
        residual_filtered_A = [a for a in A if not a.residual_arc]
        incident_vertex_set = {a.incident_vertex_ids for a in residual_filtered_A}
        new_arcs = []

        for a in residual_filtered_A:
            v1, v2 = a.incident_vertex_ids
            
            if (v2, v1) in incident_vertex_set:
                b: Optional[Arc] = next((b for b in A if b.incident_vertex_ids == (v2, v1)), None)
                if b is None: continue
                
                v3 = len(self.vertices)
                v4 = v3 + 1
                self.vertices[v3], self.vertices[v4] = Vertex(v3), Vertex(v4)
                self.connected_components.add_multi((v3, v4))
                
                a1_id = len(A) + len(new_arcs)
                a2_id, a3_id, a4_id = a1_id + 1, a1_id + 2, a1_id + 3
                new_arcs.extend([
                    Arc(a1_id, v1, v3, a.capacity, a.flow),
                    Arc(a2_id, v3, v2, a.capacity, a.flow),
                    Arc(a3_id, v2, v4, b.capacity, b.flow),
                    Arc(a4_id, v4, v1, b.capacity, b.flow),
                ])
                
                logger.warning(f"Found antiparallel arcs between {v1}-{v2} and {v2}-{v1}, constructing workaround.")
                
                A.remove(a)
                A.remove(b)
        
        residual_filtered_A.extend(new_arcs)
        return residual_filtered_A
    
    def create_residual_arcs(self, A:List[Arc]) -> List[Arc]:
        residual_arcs = []
        for a in A:
            v1, v2 = a.incident_vertex_ids
            a.id, a.capacity, a.flow, a.residual_arc
            residual_arcs.append(Arc(a.id + len(A), v2, v1, R=True))
        return residual_arcs

    def init_arcs(self, A:List[Arc]) -> None:
        A_new = self.refactor_antiparallel_arcs(A)
        residual_arcs = self.create_residual_arcs(A_new)
    
        self.init_edges(A_new)
        self.init_edges(residual_arcs)

    def update_node_levels(self) -> bool:
        if self.node_levels is None: 
            self.node_levels = {v:-1 for v in self.vertices}

        self.node_levels[self.source_node_id] = 0
        queue = deque([self.source_node_id])
        
        while queue:
            u = queue.popleft()
            for v in self.vertices[u].leafs:
                arc = self.edges[(u, v)]
                if self.node_levels[v] < 0 and arc.remaining_capacity() > 0:
                    self.node_levels[v] = self.node_levels[u] + 1
                    queue.append(v)
        
        return self.node_levels[self.sink_node_id] != -1

    def is_source_and_sink_connected(self) -> bool:
        return self.connected_components.find(self.source_node_id) == \
               self.connected_components.find(self.sink_node_id)

    def update_meta(self) -> None:
        self.direction = self.get_graph_direction()
        self.acyclical = not self.is_cyclic()
        self.st_connected = self.is_source_and_sink_connected()
    
    def initialize_flow(self) -> None:
        self.flow = 0
        for arc in self.edges.values(): 
            arc.set_flow(0)

    def augment_along(self, P:List[Tuple[int, int]], f:float):
        self.flow += f
        for u, v in P:
            self.edges[(u, v)].alter_flow(f)
            self.edges[(v, u)].alter_flow(-f)

    def copy(self):
        return Network(self.id, 
                       [v.copy() for v in self.vertices.values()], 
                       [a.copy() for a in self.edges.values()], 
                       self.source_node_id, self.sink_node_id, self.flow)
    
    def __str__(self): # TODO
        graph_info = (f"Metadata of Network{self.id}:\n"
                      + ("This is a directed " if self.direction else "This is an undirected ")
                      + ("acyclic graph " if self.acyclical else "graph that possibly includes cycles ")
                      + "with " + ("some negative " if self.has_negative_weight else "all positive ") + "edge weights\n"
                      )
        return graph_info
