from typing import List


class UnionFind:
    def __init__(self, elements:List[int]):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def add(self, element:int):
        self.parent[element] = element
        self.rank[element] = 0

    def add_multi(self, elements:List[int]):
        for element in elements:
            self.add(element)

    def find(self, element:int):
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, element1:int, element2:int):
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