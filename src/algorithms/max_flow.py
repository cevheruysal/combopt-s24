import logging

from network_flow import NetworkFlowAlgorithms
from util import find_st_path, blocking_flow
from src.enums import MaxFlowAlgorithmsEnum
from src.notation import Network

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MaxFlowAlgorithms(NetworkFlowAlgorithms):
    def __init__(self, N: Network):
        """
        Input: a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢)
        Task: compute an 𝑠-𝑡-flow 𝑓 in 𝑁 of maximum value

        An 𝑠-𝑡-flow 𝑓 in a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) is maximum if and only if there is no 𝑓-augmenting path

        Let 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) be a network, then the value of a maximum 𝑠-𝑡-flow
        is equal to the capacity of a minimum (𝑠, 𝑡)-cut in 𝑁 """

        super().__init__(N)

    def run(self, use_algorithm: MaxFlowAlgorithmsEnum = MaxFlowAlgorithmsEnum.EDMONDS_KARP) -> int:
        result = None

        if use_algorithm.value > 0:
            logger.info(f"Trying to use {use_algorithm.name} to find max flow / min cut")

            match use_algorithm.value:
                case 1:
                    result = self.ford_fulkerson_max_flow_algorithm()
                case 2:
                    result = self.edmonds_karp_max_flow_algorithm()
                case 3:
                    result = self.dinics_max_flow_algorithm()
                case _:
                    logger.error("Algorithm doesn't exist returning no solution")

        return result

    def ford_fulkerson_max_flow_algorithm(self) -> float:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) with positive arc capacities 𝑢 ∶ 𝐸(𝐺) → Q>0
        output: an 𝑠-𝑡-flow of maximum value in 𝑁

        1 initialize 𝑓 as the zero flow, i. e., 𝑓(𝑒) ∶= 0 for all 𝑒 ∈ 𝐸(𝐺)
        2 while there exists an 𝑓-augmenting 𝑠-𝑡-path in 𝑁_𝑓 do
            3 compute an 𝑠-𝑡-path 𝑃 in 𝑁_𝑓
            4 set 𝛾 ∶= min{𝑢_𝑓(𝑒) ∶ 𝑒 ∈ 𝐸(𝑃)}
            5 augment 𝑓 along 𝑃 by 𝛾
        6 return 𝑓 """

        self.network.initialize_flow()
        augmentation_count = 0

        while True:
            path, flow = find_st_path(self.network)
            if path is None or flow == 0: break
            self.network.augment_along(path, flow)
            augmentation_count += 1

        logger.info(
            f"Found maximum flow in {augmentation_count} augmentation iterations using Ford-Fulkerson: {self.network.flow}")
        return self.network.flow

    def edmonds_karp_max_flow_algorithm(self) -> float:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢) with positive arc capacities 𝑢 ∶ 𝐸(𝐺) → Q>0
        output: an 𝑠-𝑡-flow of maximum value in 𝑁

        1 initialize 𝑓 as the zero flow, i. e., 𝑓 (𝑒) ∶= 0 for all 𝑒 ∈ 𝐸(𝐺)
        2 while there exists an 𝑓-augmenting 𝑠-𝑡-path in 𝑁_𝑓 do
            3 compute an 𝑠-𝑡-path 𝑃 in 𝑁_𝑓 with a minimum number of edges
            4 set 𝛾 ∶= min{𝑢_𝑓(𝑒) ∶ 𝑒 ∈ 𝐸(𝑃)}
            5 augment 𝑓 along 𝑃 by 𝛾
        6 return 𝑓

        Regardless of the edge capacities, the Edmonds-Karp algorithm (Algorithm 5) stops after at most 𝑚*𝑛/2 augmentations.
        It can be implemented such that it computes a maximum network flow in time 𝒪(𝑚^2*𝑛) """

        self.network.initialize_flow()
        augmentation_count = 0

        while True:
            path, flow = find_st_path(self.network)
            if path is None or flow == 0: break
            self.network.augment_along(path, flow)
            augmentation_count += 1

        logger.info(
            f"Found maximum flow in {augmentation_count} augmentation iterations using Edmonds-Karp: {self.network.flow}")
        return self.network.flow

    def dinics_max_flow_algorithm(self) -> int:
        """
        input : a network 𝑁 = (𝐺, 𝑠, 𝑡, 𝑢)
        output: a maximum 𝑠-𝑡 flow 𝑓 in 𝑁
        1 initialize 𝑓 ∶= 0
        2 while there exists an 𝑓-augmenting path in 𝑁𝑓 do
            3 initialize the layered residual network 𝑁^𝐿_𝑓
            4 while there exists an 𝑠-𝑡-path in 𝑁^𝐿_𝑓 do
                5 determine a node 𝑣 ∈ 𝑉(𝑁^𝐿_𝑓 ) of minimum throughput 𝑝(𝑣)
                6 determine flow augmentation 𝑓′ through PushFlow(𝑁^𝐿_𝑓 , 𝑣, 𝑝(𝑣))
                            and PullFlow(𝑁^𝐿_𝑓 , 𝑣, 𝑝(𝑣))
                7 update 𝑓 through augmenting by 𝑓′
                8 update 𝑁^𝐿_𝑓 : update capacities and throughput,
                                 remove nodes with throughput 0,
                                 remove arcs with capacity 0
            9 determine 𝑁_𝑓 with the current flow 𝑓
        10 return 𝑓 """

        self.network.initialize_flow()

        while self.network.update_node_levels():
            start = {v: 0 for v in self.network.vertices}
            while True:
                flow = blocking_flow(self.network,
                                     self.network.source_node_id,
                                     float('inf'), start)
                if flow == 0: break
                self.network.flow += flow

        logger.info(f"Found maximum flow using Dinic's algorithm: {self.network.flow}")

        return self.network.flow
