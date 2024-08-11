from enum import Enum


class EdgeDirection(Enum):
    UNDIRECTED = 0
    DIRECTED = 1
    BIDIRECTED = 2


class GraphDirection(Enum):
    UNDIRECTED = 0
    DIRECTED = 1
    BIDIRECTED = 2
    MIXED = 3


class MinDistanceAlgorithmsEnum(Enum):
    AUTOMATIC = 0
    TOPOLOGICAL_SORT = 1
    DIJKSTRA = 2
    BELLMAN_FORD = 3
    FLOYD_WARSHALL = 4
    A_STAR = 5


class MinSpanningTreeAlgorithmsEnum(Enum):
    AUTOMATIC = 0
    PRIMS = 1
    KRUSKALS = 2


class MaxFlowAlgorithmsEnum(Enum):
    FORD_FULKERSON = 1
    EDMONDS_KARP = 2
    DINICS = 3
    PUSH_RELABEL = 4


class MinCostFlowAlgorithmsEnum(Enum):
    AUTOMATIC = 0
    MINIMUM_MEAN_CYCLE_CANCELLING = 1
    SUCCESSIVE_SHORTEST_PATH = 2
