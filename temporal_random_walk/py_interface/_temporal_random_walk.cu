#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include "../src/proxies/TemporalRandomWalk.cuh"
#include "../src/proxies/RandomPicker.cuh"
#include <stdexcept>
#include "../src/data/enums.cuh"
#include "../src/common/const.cuh"

namespace py = pybind11;

RandomPickerType picker_type_from_string(const std::string& picker_type_str)
{
    if (picker_type_str == "Uniform")
    {
        return RandomPickerType::Uniform;
    }
    else if (picker_type_str == "Linear")
    {
        return RandomPickerType::Linear;
    }
    else if (picker_type_str == "ExponentialIndex")
    {
        return RandomPickerType::ExponentialIndex;
    }
    else if (picker_type_str == "ExponentialWeight")
    {
        return RandomPickerType::ExponentialWeight;
    }
    else
    {
        throw std::invalid_argument("Invalid picker type: " + picker_type_str);
    }
}

WalkDirection walk_direction_from_string(const std::string& walk_direction_str)
{
    if (walk_direction_str == "Forward_In_Time")
    {
        return WalkDirection::Forward_In_Time;
    }
    else if (walk_direction_str == "Backward_In_Time")
    {
        return WalkDirection::Backward_In_Time;
    }
    else
    {
        throw std::invalid_argument("Invalid walk direction: " + walk_direction_str);
    }
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
                        std::optional<bool> enable_weight_computation,
                        std::optional<double> timescale_bound)
             {
                 return std::make_unique<TemporalRandomWalk>(
                     is_directed,
                     use_gpu,
                     max_time_capacity.value_or(-1),
                     enable_weight_computation.value_or(false),
                     timescale_bound.value_or(DEFAULT_TIMESCALE_BOUND));
             }),
             R"(
            Initialize a temporal random walk generator.

            Args:
            is_directed (bool): Whether to create a directed graph.
            use_gpu (bool): Whether to use GPU or not.
            max_time_capacity (int, optional): Maximum time window for edges. Edges older than (latest_time - max_time_capacity) are removed. Use -1 for no limit. Defaults to -1.
            enable_weight_computation (bool, optional): Enable CTDNE weight computation. Required for ExponentialWeight picker. Defaults to False.
            timescale_bound (float, optional): Scale factor for temporal differences. Used to prevent numerical issues with large time differences. Defaults to -1.0.
            )",
             py::arg("is_directed"),
             py::arg("use_gpu") = false,
             py::arg("max_time_capacity") = py::none(),
             py::arg("enable_weight_computation") = py::none(),
             py::arg("timescale_bound") = py::none())

        .def("add_multiple_edges", [](TemporalRandomWalk& tw, const std::vector<std::tuple<int, int, int64_t>>& edge_infos)
             {
                 tw.add_multiple_edges(edge_infos);
             },
             R"(
             Add multiple directed edges to the temporal graph.

             Args:
                edge_infos (List[Tuple[int, int, int]]): List of (source, target, timestamp) tuples.
                    Node ids (source, target) should be dense integer numbers (0 - n). Please use some label encoding beforehand.
            )",
            py::arg("edge_infos")
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
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(int))},
                    walk_set.walk_lens,
                    py::capsule(walk_set.walk_lens, [](void* p) { delete[] static_cast<int*>(p); })
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
                    py::array::StridesContainer{static_cast<ssize_t>(sizeof(int))},
                    walk_set.walk_lens,
                    py::capsule(walk_set.walk_lens, [](void* p) { delete[] static_cast<int*>(p); })
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
