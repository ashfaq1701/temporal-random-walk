import networkx as nx
from temporal_random_walk import TemporalRandomWalk
import pytest

GPU_USAGE_MODE = False

def test_networkx_integration():
    # Create a simple temporal directed graph using NetworkX
    nx_graph = nx.DiGraph()

    # Add edges with timestamps
    edges_with_timestamps = [
        (0, 1, {'timestamp': 100}),
        (1, 2, {'timestamp': 200}),
        (2, 3, {'timestamp': 300}),
        (3, 0, {'timestamp': 400}),
        (1, 3, {'timestamp': 500}),
    ]
    nx_graph.add_edges_from(edges_with_timestamps)

    # Create TemporalRandomWalk instance
    tw = TemporalRandomWalk(True, GPU_USAGE_MODE)

    # Test importing from NetworkX
    tw.add_edges_from_networkx(nx_graph)

    # Verify node count
    assert tw.get_node_count() == 4
    assert tw.get_edge_count() == 5

    # Test exporting to NetworkX
    nx_graph_2 = tw.to_networkx()

    # Verify the exported graph matches the original
    assert nx_graph_2.number_of_nodes() == nx_graph.number_of_nodes()
    assert nx_graph_2.number_of_edges() == nx_graph.number_of_edges()

    # Check if all edges and their timestamps are preserved
    for u, v, data in nx_graph.edges(data=True):
        assert nx_graph_2.has_edge(u, v)
        assert nx_graph_2[u][v]['timestamp'] == data['timestamp']

def test_networkx_integration_empty_graph():
    # Test with empty graph
    G = nx.DiGraph()
    tw = TemporalRandomWalk(True, GPU_USAGE_MODE)

    # Should work with empty graph
    tw.add_edges_from_networkx(G)
    assert tw.get_node_count() == 0
    assert tw.get_edge_count() == 0

def test_networkx_integration_invalid_timestamp():
    G = nx.DiGraph()
    # Add edge with missing timestamp
    G.add_edge(0, 1)

    tw = TemporalRandomWalk(True, GPU_USAGE_MODE)

    # Should raise an error when timestamp is missing
    with pytest.raises(KeyError):
        tw.add_edges_from_networkx(G)

def test_networkx_integration_with_existing_edges():
    # Create TemporalRandomWalk instance
    tw = TemporalRandomWalk(True, GPU_USAGE_MODE)

    # Add some initial edges directly
    initial_edges = [
        (0, 1, 50),   # Earlier timestamp
        (1, 2, 150),  # Between networkx edges
        (4, 5, 600)   # New nodes, later timestamp
    ]
    tw.add_multiple_edges(initial_edges)

    # Verify initial state
    assert tw.get_node_count() == 5  # Nodes 0,1,2,4,5
    assert tw.get_edge_count() == 3

    # Create NetworkX graph with additional edges
    nx_graph = nx.DiGraph()
    edges_with_timestamps = [
        (0, 1, {'timestamp': 100}),  # Duplicate edge, different time
        (1, 2, {'timestamp': 200}),  # Duplicate edge, different time
        (2, 3, {'timestamp': 300}),  # New edge
        (3, 0, {'timestamp': 400}),  # New edge
    ]
    nx_graph.add_edges_from(edges_with_timestamps)

    # Add edges from NetworkX
    tw.add_edges_from_networkx(nx_graph)

    # Verify updated state
    assert tw.get_node_count() == 6  # Should still include node 4,5
    assert tw.get_edge_count() == 7  # 3 initial + 4 new (including duplicates with different times)

    # Export back to NetworkX and verify
    nx_graph_2 = tw.to_networkx()

    # Verify all edges are present with correct timestamps
    expected_edges = {
        (0, 1): [50, 100],   # Both timestamps should exist
        (1, 2): [150, 200],  # Both timestamps should exist
        (2, 3): [300],       # From NetworkX only
        (3, 0): [400],       # From NetworkX only
        (4, 5): [600]        # From initial edges only
    }

    # Check if all expected edges and their timestamps exist
    for (u, v), expected_timestamps in expected_edges.items():
        edges_found = []
        # Get all edges between u and v from exported graph
        if nx_graph_2.has_edge(u, v):
            edges_found.append(nx_graph_2[u][v]['timestamp'])

        # Sort both lists for comparison
        edges_found.sort()
        expected_timestamps.sort()
        # Since we're using DiGraph, we'll only get the latest timestamp for each edge
        assert edges_found[0] == max(expected_timestamps), f"Edge ({u},{v}) has incorrect timestamp: {edges_found[0]} vs expected {max(expected_timestamps)}"

    # Verify no unexpected edges exist
    for u, v in nx_graph_2.edges():
        assert (u, v) in expected_edges, f"Unexpected edge ({u},{v}) found"

def test_networkx_integration_directed_undirected():
    # Test directed graph
    tw_directed = TemporalRandomWalk(True, GPU_USAGE_MODE)  # is_directed = True
    tw_directed.add_multiple_edges([
        (0, 1, 100),
        (1, 2, 200),
        (2, 0, 300),
    ])

    nx_directed = tw_directed.to_networkx()
    # Verify it's a directed graph
    assert isinstance(nx_directed, nx.DiGraph)
    # Check edges are directed
    assert nx_directed.has_edge(0, 1) and not nx_directed.has_edge(1, 0)
    assert nx_directed.has_edge(1, 2) and not nx_directed.has_edge(2, 1)
    assert nx_directed.has_edge(2, 0) and not nx_directed.has_edge(0, 2)

    # Test undirected graph
    tw_undirected = TemporalRandomWalk(False, GPU_USAGE_MODE)  # is_directed = False
    tw_undirected.add_multiple_edges([
        (0, 1, 100),
        (1, 2, 200),
        (2, 0, 300),
    ])

    nx_undirected = tw_undirected.to_networkx()
    # Verify it's an undirected graph
    assert isinstance(nx_undirected, nx.Graph)
    # Check edges are undirected (both directions exist)
    assert nx_undirected.has_edge(0, 1) and nx_undirected.has_edge(1, 0)
    assert nx_undirected.has_edge(1, 2) and nx_undirected.has_edge(2, 1)
    assert nx_undirected.has_edge(2, 0) and nx_undirected.has_edge(0, 2)

    # Verify timestamps are preserved
    assert nx_directed[0][1]['timestamp'] == 100
    assert nx_directed[1][2]['timestamp'] == 200
    assert nx_directed[2][0]['timestamp'] == 300

    assert nx_undirected[0][1]['timestamp'] == 100
    assert nx_undirected[1][2]['timestamp'] == 200
    assert nx_undirected[2][0]['timestamp'] == 300


if __name__ == "__main__":
    test_networkx_integration()
    test_networkx_integration_empty_graph()
    test_networkx_integration_invalid_timestamp()
    test_networkx_integration_with_existing_edges()
    test_networkx_integration_directed_undirected()
    print("All tests passed!")
