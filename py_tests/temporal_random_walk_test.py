import csv
import time

from temporal_random_walk import TemporalRandomWalk


GPU_USAGE_MODE = False

def print_walks_for_nodes(walks):
    for idx, walk in enumerate(walks):
        nodes, timestamps = walk
        print(f"Walk {idx}: {','.join(map(str, nodes))}, Timestamps: {','.join(map(str, timestamps))}")

if __name__ == '__main__':
    temporal_random_walk_obj = TemporalRandomWalk(
        is_directed=False,
        use_gpu=GPU_USAGE_MODE,
        max_time_capacity=-1,
        enable_weight_computation=False
    )

    data_file_path = '../data/sample_data.csv'

    data_tuples = []

    with open(data_file_path, mode='r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            u = int(row[0])
            i = int(row[1])
            ts = int(row[2])
            data_tuples.append((u, i, ts))

    print(f"--- Total edges: {len(data_tuples)} ---")

    start_time = time.time()
    temporal_random_walk_obj.add_multiple_edges(data_tuples)
    print(f"--- Edge addition time : {time.time() - start_time}---")

    print(f'Total edges: {temporal_random_walk_obj.get_edge_count()}')
    print(f'Total nodes: {temporal_random_walk_obj.get_node_count()}')

    start_time = time.time()
    walks_1 = temporal_random_walk_obj.get_random_walks_and_times(
        max_walk_len=20,
        walk_bias="ExponentialIndex",
        num_walks_total=100_000,
        initial_edge_bias="Uniform",
        walk_direction="Forward_In_Time"
    )

    walks_2 = temporal_random_walk_obj.get_random_walks_and_times(
        max_walk_len=20,
        walk_bias="ExponentialIndex",
        num_walks_total=100_000,
        initial_edge_bias="Uniform",
        walk_direction="Forward_In_Time"
    )
    print(f"--- Walk generation time : {time.time() - start_time}---")

    print(f"Number of walks {len(walks_2)}")
    print_walks_for_nodes(walks_2[:100])
