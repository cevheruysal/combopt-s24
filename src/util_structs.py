from typing import Dict, List, Optional, Tuple


class UnionFind:
    __slots__ = ["parent", "rank"]

    def __init__(self, elements: List[int]):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def add(self, element: int):
        """
        add new element to the UnionFind  object 
        set its parent as itself and set its rank as 0 
        """
        self.parent[element] = element
        self.rank[element] = 0

    def add_multi(self, elements: List[int]):
        """
        add multiple elements stored in a iterable object
        """
        for element in elements:
            self.add(element)

    def find(self, element: int):
        """
        search the parent of an element 
        until the parents parent is itself
        """
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, element1: int, element2: int):
        """
        place two items under the same parent item
        """
        root1 = self.find(element1)
        root2 = self.find(element2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

class VertexPropItem:
    __slots__ = ["dist", "prev", "in_deg", "level", "comp"]
    def __init__(self, 
                 d:float = float("inf"), p:Optional[int] = None):
        self.dist = d
        self.prev = p

class VertexProp:
    __slots__ = ["vertices"]
    def __init__(self, V:List[int], S:int = -1):
        self.vertices:Dict[int, VertexPropItem] = {v: VertexPropItem(d=0) if v == S 
                                                   else VertexPropItem() for v in V}
    
    def get_dist(self, v:int) -> float:
        if v not in self.vertices:
            return KeyError("Cannot get distance of index, key is not found")
        return self.vertices[v].dist
    
    def get_prev(self, v:int) -> int:
        if v not in self.vertices:
            return KeyError("Cannot get the previous, key is not found")
        return self.vertices[v].prev
    
    def set_dist(self, v:int, d:float) -> None:
        if v not in self.vertices:
            return KeyError("Cannot set distance of index, key is not found")
        self.vertices[v].dist = d

    def set_prev(self, v:int, p:int) -> None:
        if v not in self.vertices:
            return KeyError("Cannot set the previous index, key is not found")
        self.vertices[v].prev = p
        
    def construct_path_to_node(self, S:int, T:int) -> List[Tuple[int, int]]:
        path = []
        node1, node2 = S, T

        while node2 != node1:
            path.insert(0, (self.get_prev(node2), node2))
            node2 = self.get_prev(node2)
        return path
    

class FCostProp:
    __slots__ = ["steps_vertices"]
    def __init__(self, V:List[int], S:int):
        self.steps_vertices: List[Dict[int, VertexPropItem]] = [{v: VertexPropItem() for v in V} 
                                                                                    for k in V]
        self.steps_vertices[0][S].dist = 0

    def get_score(self, k:int, v:int) -> float:
        if v not in self.steps_vertices[k]:
            return KeyError("Cannot get distance of index, key is not found")
        return self.steps_vertices[k][v].dist
    
    def get_prev(self, k:int, v:int) -> int:
        if v not in self.steps_vertices[k]:
            return KeyError("Cannot get the previous, key is not found")
        return self.steps_vertices[k][v].prev
    
    def set_score(self, k:int, v:int, d:float) -> None:
        if v not in self.steps_vertices[k]:
            return KeyError("Cannot set distance of index, key is not found")
        self.steps_vertices[k][v].dist = d

    def set_prev(self, k:int, v:int, p:int) -> None:
        if v not in self.steps_vertices[k]:
            return KeyError("Cannot set the previous index, key is not found")
        self.steps_vertices[k][v].prev = p

    def get_mu(self, k:int, x:int) -> float:
        n = len(self.steps_vertices)
        return (self.get_score(n, x) - self.get_score(k, x)) / (n - k)
        
    def construct_path_to_node(self, S:int, T:int) -> List[Tuple[int, int]]:
        path = []
        node1, node2 = S, T

        while node2 != node1:
            path.insert(0, (self.get_prev(node2), node2))
            node2 = self.get_prev(node2)
        return path
