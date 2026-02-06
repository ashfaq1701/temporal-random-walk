#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include "../src/proxies/TemporalRandomWalk.cuh"
#include "../src/proxies/RandomPicker.cuh"
#include <stdexcept>
#include "../src/data/enums.cuh"
#include "../src/common/const.cuh"
#include "../src/core/helpers.cuh"

namespace py = pybind11;

inline double resolve_temporal_node2vec_parameter(
    const std::optional<double>& temporal_param,
    const double default_value,
    const char* temporal_name) {

    const double value = temporal_param.value_or(default_value);

    if (value <= 0.0) {
        throw std::invalid_argument(
            std::string("'") + temporal_name + "' must be > 0.");
    }

    return value;
}

inline const RandomPickerType* get_picker_ptr_from_optional_string(
    const std::optional<std::string>& picker_str,
    std::optional<RandomPickerType>& picker_enum_storage) {

    if (picker_str.has_value()) {
        picker_enum_storage = picker_type_from_string(*picker_str);
        return &(*picker_enum_storage);
    }
    return nullptr;
}

PYBIND11_MODULE(_temporal_random_walk, m)
{
    py::class_<TemporalRandomWalk>(m, "TemporalRandomWalk")
        .def(py::init([](const bool is_directed, bool use_gpu, const std::optional<int64_t> max_time_capacity,
                         const std::optional<bool> enable_weight_computation, const std::optional<double> timescale_bound,
                         const std::optional<double> temporal_node2vec_p,
                         const std::optional<double> temporal_node2vec_q,
                         const std::optional<int> walk_padding_value,
                         const std::optional<uint64_t> global_seed)
             {
                 const double resolved_node2vec_p = resolve_temporal_node2vec_parameter(
                     temporal_node2vec_p,
                     DEFAULT_NODE2VEC_P,
                     "temporal_node2vec_p");

                 const double resolved_node2vec_q = resolve_temporal_node2vec_parameter(
                     temporal_node2vec_q,
                     DEFAULT_NODE2VEC_Q,
                     "temporal_node2vec_q");

                 return std::make_unique<TemporalRandomWalk>(
                     is_directed,
                     use_gpu,
                     max_time_capacity.value_or(-1),
                     enable_weight_computation.value_or(false),
                     timescale_bound.value_or(DEFAULT_TIMESCALE_BOUND),
                     resolved_node2vec_p,
                     resolved_node2vec_q,
                     walk_padding_value.value_or(EMPTY_NODE_VALUE),
                     global_seed.value_or(EMPTY_GLOBAL_SEED));
             }),
             R"(
             Initialize a temporal random walk generator.

             Args:
             is_directed (bool): Whether to create a directed graph.
             use_gpu (bool): Whether to use GPU or not.
             max_time_capacity (int, optional): Maximum time window for edges. Edges older than (latest_time - max_time_capacity) are removed. Use -1 for no limit. Defaults to -1.
             enable_weight_computation (bool, optional): Enable CTDNE weight computation. Required for ExponentialWeight picker. Defaults to False.
             timescale_bound (float, optional): Scale factor for temporal differences. Used to prevent numerical issues with large time differences. Defaults to -1.0.
             temporal_node2vec_p (float, optional): Temporal-node2vec return parameter p (> 0). Defaults to 1.0.
             temporal_node2vec_q (float, optional): Temporal-node2vec in-out parameter q (> 0). Defaults to 1.0.
             walk_padding_value (int, optional): Padding node value for prematurely broken walks. Default is -1.
             global_seed (int, optional): A global seed to have reproducibility inside random walks. Default is empty and the code in that case generates random seed in each run.
             )",
             py::arg("is_directed"),
             py::arg("use_gpu") = false,
             py::arg("max_time_capacity") = py::none(),
             py::arg("enable_weight_computation") = py::none(),
             py::arg("timescale_bound") = py::none(),
             py::arg("temporal_node2vec_p") = py::none(),
             py::arg("temporal_node2vec_q") = py::none(),
             py::arg("walk_padding_value") = py::none(),
             py::arg("global_seed") = py::none())

        .def("add_multiple_edges", [](TemporalRandomWalk& tw,
                             const py::array_t<int>& sources,
                             const py::array_t<int>& targets,
                             const py::array_t<int64_t>& timestamps)
        {
            // Request buffer access to numpy arrays
            const auto sources_info = sources.request();
            const auto targets_info = targets.request();
            const auto timestamps_info = timestamps.request();

            // Check dimensions and sizes
            if (sources_info.ndim != 1 || targets_info.ndim != 1 || timestamps_info.ndim != 1)
                throw std::runtime_error("Arrays must be 1-dimensional");

            // Check that all arrays have the same length
            const size_t num_edges = sources_info.shape[0];
            if (targets_info.shape[0] != num_edges || timestamps_info.shape[0] != num_edges)
                throw std::runtime_error("All arrays must have the same length");

            // Get pointers to the data
            const auto sources_ptr = static_cast<int*>(sources_info.ptr);
            const auto targets_ptr = static_cast<int*>(targets_info.ptr);
            const auto* timestamps_ptr = static_cast<int64_t*>(timestamps_info.ptr);

            // Call the C++ function with raw pointers and size
            tw.add_multiple_edges(sources_ptr, targets_ptr, timestamps_ptr, num_edges);
        },
        R"(
        Add multiple directed edges to the temporal graph.

        This function efficiently handles both Python lists and NumPy arrays without
        unnecessary data copying. The implementation automatically converts the input
        data to the appropriate C++ types.

        Args:
            sources: List or NumPy array of source node IDs (or first node in undirected graphs).
            targets: List or NumPy array of target node IDs (or second node in undirected graph).
            timestamps: List or NumPy array of timestamps representing when interactions occurred.

        Raises:
            RuntimeError: If arrays are not 1-dimensional or have different lengths.

        Note:
            For large datasets, NumPy arrays will provide better performance than Python lists.
        )",
        py::arg("sources"),
        py::arg("targets"),
        py::arg("timestamps")
        )
        .def("get_random_walks_and_times_for_all_nodes", [](TemporalRandomWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const int num_walks_per_node,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time")
            {
                const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                std::optional<RandomPickerType> edge_bias_enum_opt;
                const RandomPickerType* initial_edge_bias_enum_ptr = get_picker_ptr_from_optional_string(
                    initial_edge_bias, edge_bias_enum_opt);
                const WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                WalkSet walk_set = tw.get_random_walks_and_times_for_all_nodes(
                    max_walk_len,
                    &walk_bias_enum,
                    num_walks_per_node,
                    initial_edge_bias_enum_ptr,
                    walk_direction_enum);

                py::array_t nodes_array(
                    py::array::ShapeContainer{static_cast<ssize_t>(walk_set.num_walks), static_cast<ssize_t>(max_walk_len)},
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(int) * max_walk_len), static_cast<ssize_t>(sizeof(int))},
                    walk_set.nodes,
                    py::capsule(walk_set.nodes, [](void* p) { delete[] static_cast<int*>(p); }));

                py::array_t timestamps_array(
                    py::array::ShapeContainer{static_cast<ssize_t>(walk_set.num_walks), static_cast<ssize_t>(max_walk_len)},
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(int64_t) * max_walk_len), static_cast<ssize_t>(sizeof(int64_t))},
                    walk_set.timestamps,
                    py::capsule(walk_set.timestamps, [](void* p) { delete[] static_cast<int64_t*>(p); }));

                py::array_t lens_array(
                    py::array::ShapeContainer{static_cast<ssize_t>(walk_set.num_walks)},
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(size_t))},
                    walk_set.walk_lens,
                    py::capsule(walk_set.walk_lens, [](void* p) { delete[] static_cast<size_t*>(p); })
                );

                walk_set.owns_data = false;

                return std::make_tuple(nodes_array, timestamps_array, lens_array);
            },
            R"(
            Generate temporal random walks with timestamps starting from all nodes.

            Args:
                max_walk_len (int): Maximum length of each random walk.
                walk_bias (str): Type of bias for selecting next node.
                    Choices:
                        - "Uniform": Equal probability
                        - "Linear": Linear time decay
                        - "ExponentialIndex": Exponential decay with indices
                        - "ExponentialWeight": Exponential decay with weights
                        - "TemporalNode2Vec": Temporal-node2vec transition bias
                num_walks_per_node (int): Number of walks per starting node.
                initial_edge_bias (str, optional): Bias type for first edge selection.
                    Uses walk_bias if not specified.
                walk_direction (str, optional): Direction of temporal random walks.
                    Either "Forward_In_Time" (default) or "Backward_In_Time".

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - 2D array of node ids (shape: [num_walks, max_walk_len])
                    - 2D array of timestamps (shape: [num_walks, max_walk_len])
                    - 1D array of actual walk lengths (shape: [num_walks])
            )",
            py::arg("max_walk_len"),
            py::arg("walk_bias"),
            py::arg("num_walks_per_node"),
            py::arg("initial_edge_bias") = py::none(),
            py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks_and_times", [](TemporalRandomWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const int num_walks_total,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time")
            {
                const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                std::optional<RandomPickerType> edge_bias_enum_opt;
                const RandomPickerType* initial_edge_bias_enum_ptr = get_picker_ptr_from_optional_string(
                    initial_edge_bias, edge_bias_enum_opt);

                const WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                WalkSet walk_set = tw.get_random_walks_and_times(
                    max_walk_len,
                    &walk_bias_enum,
                    num_walks_total,
                    initial_edge_bias_enum_ptr,
                    walk_direction_enum);

                py::array_t nodes_array(
                    py::array::ShapeContainer{static_cast<ssize_t>(walk_set.num_walks), static_cast<ssize_t>(max_walk_len)},
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(int) * max_walk_len), static_cast<ssize_t>(sizeof(int))},
                    walk_set.nodes,
                    py::capsule(walk_set.nodes, [](void* p) { delete[] static_cast<int*>(p); }));

                py::array_t timestamps_array(
                    py::array::ShapeContainer{static_cast<ssize_t>(walk_set.num_walks), static_cast<ssize_t>(max_walk_len)},
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(int64_t) * max_walk_len), static_cast<ssize_t>(sizeof(int64_t))},
                    walk_set.timestamps,
                    py::capsule(walk_set.timestamps, [](void* p) { delete[] static_cast<int64_t*>(p); }));

                py::array_t lens_array(
                    py::array::ShapeContainer{static_cast<ssize_t>(walk_set.num_walks)},
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(size_t))},
                    walk_set.walk_lens,
                    py::capsule(walk_set.walk_lens, [](void* p) { delete[] static_cast<size_t*>(p); })
                );

                walk_set.owns_data = false;

                return std::make_tuple(nodes_array, timestamps_array, lens_array);
            },
            R"(
            Generate temporal random walks with timestamps.

            Args:
                max_walk_len (int): Maximum length of each random walk.
                walk_bias (str): Type of bias for selecting next node.
                    Choices:
                        - "Uniform": Equal probability for all edges
                        - "Linear": Linear time decay
                        - "ExponentialIndex": Exponential decay with indices
                        - "ExponentialWeight": Exponential decay with weights
                        - "TemporalNode2Vec": Temporal-node2vec transition bias
                num_walks_total (int): Total Number of walks to generate.
                initial_edge_bias (str, optional): Bias type for first edge selection.
                    Uses walk_bias if not specified.
                walk_direction (str, optional): Direction of temporal random walks.
                    Either "Forward_In_Time" (default) or "Backward_In_Time".

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - 2D array of node ids (shape: [num_walks, max_walk_len])
                    - 2D array of timestamps (shape: [num_walks, max_walk_len])
                    - 1D array of actual walk lengths (shape: [num_walks])
            )",
            py::arg("max_walk_len"),
            py::arg("walk_bias"),
            py::arg("num_walks_total"),
            py::arg("initial_edge_bias") = py::none(),
            py::arg("walk_direction") = "Forward_In_Time")

        .def("get_node_count", &TemporalRandomWalk::get_node_count,
            R"(
            Get total number of nodes in the graph.

            Returns:
                int: Number of active nodes.
            )")

        .def("get_edge_count", &TemporalRandomWalk::get_edge_count,
             R"(
             Returns the total number of directed edges in the temporal graph.

             Returns:
                int: The total number of directed edges.
             )")

        .def("get_node_ids", [](const TemporalRandomWalk& tw)
             {
                 const auto& node_ids = tw.get_node_ids();
                 py::array_t<int> py_node_ids(static_cast<long>(node_ids.size()));

                 auto py_node_ids_mutable = py_node_ids.mutable_unchecked<1>();
                 for (size_t i = 0; i < node_ids.size(); ++i)
                 {
                     py_node_ids_mutable(i) = node_ids[i];
                 }

                 return py_node_ids;
             },
             R"(
             Returns a NumPy array containing the IDs of all nodes in the temporal graph.

            Returns:
                np.ndarray: A NumPy array with all node IDs.
            )"
        )

        .def("clear", &TemporalRandomWalk::clear,
             R"(
            Clears and reinitiates the underlying graph.
            )"
        )

        .def("get_memory_used", &TemporalRandomWalk::get_memory_used,
            R"(
            Returns the memory used by the application in bytes.

            Returns:
                int: The total number of bytes allocated.
            )"
        )

        .def("add_edges_from_networkx", [](TemporalRandomWalk& tw, const py::object& nx_graph)
             {
                 const py::object edges = nx_graph.attr("edges")(py::arg("data") = true);

                 std::vector<std::tuple<int, int, int64_t>> edge_infos;
                 for (const auto& edge : edges)
                 {
                     auto edge_tuple = edge.cast<py::tuple>();
                     const int source = py::cast<int>(edge_tuple[0]);
                     const int target = py::cast<int>(edge_tuple[1]);
                     const auto attrs = edge_tuple[2].cast<py::dict>();
                     const int64_t timestamp = py::cast<int64_t>(attrs["timestamp"]);

                     edge_infos.emplace_back(source, target, timestamp);
                 }

                 tw.add_multiple_edges(edge_infos);
             },
             R"(
            Add edges from a NetworkX graph.

            Args:
                nx_graph (networkx.Graph): NetworkX graph with timestamp edge attributes.
            )"
        )

        .def("to_networkx", [](TemporalRandomWalk& tw)
             {
                 const auto edges = tw.get_edges();

                 const py::module nx = py::module::import("networkx");
                 const py::object GraphClass = tw.get_is_directed() ? nx.attr("DiGraph") : nx.attr("Graph");
                 py::object nx_graph = GraphClass();

                 for (const auto& [src, dest, ts] : edges)
                 {
                     py::dict kwargs;
                     kwargs["timestamp"] = ts;

                     nx_graph.attr("add_edge")(src, dest, **kwargs);
                 }

                 return nx_graph;
             },
             R"(
             Export graph to NetworkX format.

            Returns:
                networkx.Graph: NetworkX graph with timestamp edge attributes.
            )"
        );

    py::class_<LinearRandomPicker>(m, "LinearRandomPicker")
        .def(py::init([](const bool use_gpu)
             {
                 return LinearRandomPicker(use_gpu);
             }),
             R"(
            Initialize linear time decay random picker.

            Args:
                use_gpu (bool): Should use GPU or not.
            )",
             py::arg("use_gpu") = false)

        .def("pick_random", &LinearRandomPicker::pick_random,
            R"(
            Pick random index with linear time decay probability.

            Args:
                start (int): Start index inclusive
                end (int): End index exclusive
                prioritize_end (bool, optional): Prioritize recent timestamps. Default: True

            Returns:
                int: Selected index
            )",
            py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<ExponentialIndexRandomPicker>(m, "ExponentialIndexRandomPicker")
        .def(py::init([](const bool use_gpu)
             {
                 return ExponentialIndexRandomPicker(use_gpu);
             }),
             R"(
            Initialize index based exponential time decay random picker.

            Args:
                Args:
                use_gpu (bool): Should use GPU or not.
            )",
             py::arg("use_gpu") = false)

        .def("pick_random", &ExponentialIndexRandomPicker::pick_random,
            R"(
            Pick random index with index based exponential time decay probability.

            Args:
                start (int): Start index inclusive
                end (int): End index exclusive
                prioritize_end (bool, optional): Prioritize recent timestamps. Default: True

            Returns:
                int: Selected index
            )",
            py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<UniformRandomPicker>(m, "UniformRandomPicker")
        .def(py::init([](const bool use_gpu)
             {
                 return UniformRandomPicker(use_gpu);
             }),
             R"(
            Initialize uniform random picker.

            Args:
                use_gpu (bool): Should use GPU or not.
            )",
             py::arg("use_gpu") = false)

        .def("pick_random", &UniformRandomPicker::pick_random,
            R"(
            Pick random index with uniform probability.

            Args:
                start (int): Start index inclusive
                end (int): End index exclusive
                prioritize_end (bool, optional): Prioritize recent timestamps. Default: True

            Returns:
                int: Selected index
            )",
            py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<WeightBasedRandomPicker>(m, "WeightBasedRandomPicker")
        .def(py::init([](const bool use_gpu)
            {
                return WeightBasedRandomPicker(use_gpu);
            }),
            R"(
            Initialize weight-based exponential time decay random picker.

            Args:
                use_gpu (bool): Should use GPU or not.
            )",
            py::arg("use_gpu") = false)

        .def("pick_random", [](const WeightBasedRandomPicker& picker, const std::vector<double>& cumulative_weights,
            const int group_start, const int group_end)
            {
                return picker.pick_random(cumulative_weights, group_start, group_end);
            },
            R"(
            Pick random index with exponential weight-based probability using cumulative weights.

            Args:
                cumulative_weights (List[float]): List of cumulative weights
                group_start (int): Start index inclusive
                group_end (int): End index exclusive

            Returns:
                int: Selected index
        )",
        py::arg("cumulative_weights"), py::arg("group_start"), py::arg("group_end"));
}
