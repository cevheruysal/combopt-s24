from src.notation import Network, Graph


class NetworkFlowAlgorithms:
    def __init__(self, N: Network):
        self.network = N
        self.residual_network = N.copy()

    def run(self) -> Graph:
        pass
