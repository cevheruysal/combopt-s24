import logging

from config import ROUND_TO
from graph_utils import generate_random_directed_connected_graph
from graph_algorithms import MinDistanceAlgorithms


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


G = generate_random_directed_connected_graph(0, 10, 30, (0, 10))
G.disp()

start_vertex_id = 0
min_dist = MinDistanceAlgorithms(G)
distances_from_start_vertex = min_dist.run(start_vertex_id, 99)
print({key:round(value,ROUND_TO) for key, value in distances_from_start_vertex.items()} 
      if distances_from_start_vertex is not None else "")