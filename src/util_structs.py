from typing import List


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
