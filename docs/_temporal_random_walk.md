Module _temporal_random_walk
============================

Classes
-------

`ExponentialIndexRandomPicker(...)`
:   __init__(self: _temporal_random_walk.ExponentialIndexRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize index based exponential time decay random picker.
    
    Args:
        Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_random_walk.ExponentialIndexRandomPicker, start: int, end: int, prioritize_end: bool = True)`
    :   Pick random index with index based exponential time decay probability.
        
        Args:
            start (int): Start index inclusive
            end (int): End index exclusive
            prioritize_end (bool, optional): Prioritize recent timestamps. Default: True
        
        Returns:
            int: Selected index

`LinearRandomPicker(...)`
:   __init__(self: _temporal_random_walk.LinearRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize linear time decay random picker.
    
    Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_random_walk.LinearRandomPicker, start: int, end: int, prioritize_end: bool = True)`
    :   Pick random index with linear time decay probability.
        
        Args:
            start (int): Start index inclusive
            end (int): End index exclusive
            prioritize_end (bool, optional): Prioritize recent timestamps. Default: True
        
        Returns:
            int: Selected index

`TemporalRandomWalk(...)`
:   __init__(self: _temporal_random_walk.TemporalRandomWalk, is_directed: bool, use_gpu: bool = False, max_time_capacity: Optional[int] = None, enable_weight_computation: Optional[bool] = None, timescale_bound: Optional[float] = None) -> None
    
    
    Initialize a temporal random walk generator.
    
    Args:
    is_directed (bool): Whether to create a directed graph.
    use_gpu (bool): Whether to use GPU or not.
    max_time_capacity (int, optional): Maximum time window for edges. Edges older than (latest_time - max_time_capacity) are removed. Use -1 for no limit. Defaults to -1.
    enable_weight_computation (bool, optional): Enable CTDNE weight computation. Required for ExponentialWeight picker. Defaults to False.
    timescale_bound (float, optional): Scale factor for temporal differences. Used to prevent numerical issues with large time differences. Defaults to -1.0.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `add_edges_from_networkx(self: _temporal_random_walk.TemporalRandomWalk, arg0: object)`
    :   Add edges from a NetworkX graph.
        
        Args:
            nx_graph (networkx.Graph): NetworkX graph with timestamp edge attributes.

    `add_multiple_edges(self: _temporal_random_walk.TemporalRandomWalk, edge_infos: list[tuple[int, int, int]])`
    :   Add multiple directed edges to the temporal graph.
        
        Args:
           edge_infos (List[Tuple[int, int, int]]): List of (source, target, timestamp) tuples.

    `clear(self: _temporal_random_walk.TemporalRandomWalk)`
    :   Clears and reinitiates the underlying graph.

    `get_edge_count(self: _temporal_random_walk.TemporalRandomWalk)`
    :   Returns the total number of directed edges in the temporal graph.
        
        Returns:
           int: The total number of directed edges.

    `get_node_count(self: _temporal_random_walk.TemporalRandomWalk)`
    :   Get total number of nodes in the graph.
        
        Returns:
            int: Number of active nodes.

    `get_node_ids(self: _temporal_random_walk.TemporalRandomWalk)`
    :   get_node_ids(self: _temporal_random_walk.TemporalRandomWalk) -> numpy.ndarray[numpy.int32]
        
        
         Returns a NumPy array containing the IDs of all nodes in the temporal graph.
        
        Returns:
            np.ndarray: A NumPy array with all node IDs.

    `get_random_walks(self: _temporal_random_walk.TemporalRandomWalk, max_walk_len: int, walk_bias: str, num_walks_total: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generates temporal random walks.
        
        Args:
            max_walk_len (int): Maximum length of each random walk
            walk_bias (str): Type of bias for selecting next edges during walk.
                Can be one of:
                    - "Uniform": Equal probability for all valid edges
                    - "Linear": Linear decay based on time
                    - "ExponentialIndex": Exponential decay with index sampling
                    - "ExponentialWeight": Exponential decay with timestamp weights
            num_walks_total (int): Total number of walks to generate.
            initial_edge_bias (str, optional): Bias type for selecting first edge.
                Uses walk_bias if not specified.
            walk_direction (str, optional): Direction of temporal random walk.
                Either "Forward_In_Time" (default) or "Backward_In_Time"
        
        Returns:
            List[List[int]]: A list of walks, where each walk is a list of node IDs
                representing a temporal path through the network.

    `get_random_walks_and_times(self: _temporal_random_walk.TemporalRandomWalk, max_walk_len: int, walk_bias: str, num_walks_total: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generate temporal random walks with timestamps.
        
        Args:
            max_walk_len (int): Maximum length of each random walk.
            walk_bias (str): Type of bias for selecting next node.
                Choices:
                    - "Uniform": Equal probability for all edges
                    - "Linear": Linear time decay
                    - "ExponentialIndex": Exponential decay with indices
                    - "ExponentialWeight": Exponential decay with weights
            num_walks_total (int): Total Number of walks to generate.
            initial_edge_bias (str, optional): Bias type for first edge selection.
                Uses walk_bias if not specified.
            walk_direction (str, optional): Direction of temporal random walks.
                Either "Forward_In_Time" (default) or "Backward_In_Time".
        
        Returns:
            List[List[Tuple[int, int]]]: List of walks where each walk is a sequence of
                (node_id, timestamp) pairs representing temporal paths through the network.

    `get_random_walks_and_times_for_all_nodes(self: _temporal_random_walk.TemporalRandomWalk, max_walk_len: int, walk_bias: str, num_walks_per_node: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generate temporal random walks with timestamps starting from all nodes.
        
        Args:
            max_walk_len (int): Maximum length of each random walk.
            walk_bias (str): Type of bias for selecting next node.
                Choices:
                    - "Uniform": Equal probability
                    - "Linear": Linear time decay
                    - "ExponentialIndex": Exponential decay with indices
                    - "ExponentialWeight": Exponential decay with weights
            num_walks_per_node (int): Number of walks per starting node.
            initial_edge_bias (str, optional): Bias type for first edge selection.
                Uses walk_bias if not specified.
            walk_direction (str, optional): Direction of temporal random walks.
                Either "Forward_In_Time" (default) or "Backward_In_Time".
        
        Returns:
            List[List[Tuple[int, int]]]: List of walks as (node_id, timestamp) sequences.

    `get_random_walks_for_all_nodes(self: _temporal_random_walk.TemporalRandomWalk, max_walk_len: int, walk_bias: str, num_walks_per_node: int, initial_edge_bias: str | None = None, walk_direction: str = 'Forward_In_Time')`
    :   Generate temporal random walks starting from all nodes.
        
        Args:
            max_walk_len (int): Maximum length of each random walk.
            walk_bias (str): Type of bias for selecting next node.
                Choices:
                    - "Uniform": Equal probability
                    - "Linear": Linear time decay
                    - "ExponentialIndex": Exponential decay with indices
                    - "ExponentialWeight": Exponential decay with weights
            num_walks_per_node (int): Number of walks per starting node.
            initial_edge_bias (str, optional): Bias type for first edge selection.
                Uses walk_bias if not specified.
            walk_direction (str, optional): Direction of temporal random walks.
                Either "Forward_In_Time" (default) or "Backward_In_Time".
        
        Returns:
            List[List[int]]: List of walks as node ID sequences.

    `to_networkx(self: _temporal_random_walk.TemporalRandomWalk)`
    :   Export graph to NetworkX format.
        
        Returns:
            networkx.Graph: NetworkX graph with timestamp edge attributes.

`UniformRandomPicker(...)`
:   __init__(self: _temporal_random_walk.UniformRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize uniform random picker.
    
    Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_random_walk.UniformRandomPicker, start: int, end: int, prioritize_end: bool = True)`
    :   Pick random index with uniform probability.
        
        Args:
            start (int): Start index inclusive
            end (int): End index exclusive
            prioritize_end (bool, optional): Prioritize recent timestamps. Default: True
        
        Returns:
            int: Selected index

`WeightBasedRandomPicker(...)`
:   __init__(self: _temporal_random_walk.WeightBasedRandomPicker, use_gpu: bool = False) -> None
    
    
    Initialize weight-based exponential time decay random picker.
    
    Args:
        use_gpu (bool): Should use GPU or not.

    ### Ancestors (in MRO)

    * pybind11_builtins.pybind11_object

    ### Methods

    `pick_random(self: _temporal_random_walk.WeightBasedRandomPicker, cumulative_weights: list[float], group_start: int, group_end: int)`
    :   Pick random index with exponential weight-based probability using cumulative weights.
        
        Args:
            cumulative_weights (List[float]): List of cumulative weights
            group_start (int): Start index inclusive
            group_end (int): End index exclusive
        
        Returns:
            int: Selected index