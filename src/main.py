import logging

from config import ROUND_TO
from graph_utils import generate_random_directed_connected_graph
from algorithms.min_distance import MinDistanceAlgorithmSolver

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


G = generate_random_directed_connected_graph(0, 10, 30, (0, 10))
G.disp()

start_vertex_id = 0
min_dist = MinDistanceAlgorithmSolver(G)
distances_from_start_vertex = min_dist.run(start_vertex_id, 99)
distances_from_start_vertex = {key: round(distances_from_start_vertex[key], ROUND_TO) for key in distances_from_start_vertex}\
    if distances_from_start_vertex else {}
print(distances_from_start_vertex)
