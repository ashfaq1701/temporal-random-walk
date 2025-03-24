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

PYBIND11_MODULE(_temporal_random_walk, m)
{
    py::class_<TemporalRandomWalk>(m, "TemporalRandomWalk")
        .def(py::init([](const bool is_directed, bool use_gpu, const std::optional<int64_t> max_time_capacity,
                        std::optional<bool> enable_weight_computation,
                        std::optional<double> timescale_bound,
                        std::optional<int> node_count_max_bound)
             {
                 return std::make_unique<TemporalRandomWalk>(
                     is_directed,
                     use_gpu,
                     max_time_capacity.value_or(-1),
                     enable_weight_computation.value_or(false),
                     timescale_bound.value_or(DEFAULT_TIMESCALE_BOUND),
                     node_count_max_bound.value_or(DEFAULT_NODE_COUNT_MAX_BOUND));
             }),
             R"(
            Initialize a temporal random walk generator.

            Args:
            is_directed (bool): Whether to create a directed graph.
            use_gpu (bool): Whether to use GPU or not.
            max_time_capacity (int, optional): Maximum time window for edges. Edges older than (latest_time - max_time_capacity) are removed. Use -1 for no limit. Defaults to -1.
            enable_weight_computation (bool, optional): Enable CTDNE weight computation. Required for ExponentialWeight picker. Defaults to False.
            timescale_bound (float, optional): Scale factor for temporal differences. Used to prevent numerical issues with large time differences. Defaults to -1.0.
            node_count_max_bound (int, optional): Maximum node count in the graph. Defaults to 10000. Setting this to a realistically higher bound can help save memory.
            )",
             py::arg("is_directed"),
             py::arg("use_gpu") = false,
             py::arg("max_time_capacity") = py::none(),
             py::arg("enable_weight_computation") = py::none(),
             py::arg("timescale_bound") = py::none(),
             py::arg("node_count_max_bound") = py::none())
        .def("add_multiple_edges", [](TemporalRandomWalk& tw, const std::vector<std::tuple<int, int, int64_t>>& edge_infos)
             {
                 tw.add_multiple_edges(edge_infos);
             },
             R"(
             Add multiple directed edges to the temporal graph.

             Args:
                edge_infos (List[Tuple[int, int, int]]): List of (source, target, timestamp) tuples.
            )",
            py::arg("edge_infos")
        )
        .def("get_random_walks_for_all_nodes", [](TemporalRandomWalk& tw,
                                                  const int max_walk_len,
                                                  const std::string& walk_bias,
                                                  const int num_walks_per_node,
                                                  const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                                  const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 const WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 return tw.get_random_walks_for_all_nodes(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_per_node,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);
             },
             R"(
             Generate temporal random walks starting from all nodes.

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
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_walks_per_node"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks_and_times_for_all_nodes", [](TemporalRandomWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const int num_walks_total,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 const WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 const auto walks_with_times = tw.get_random_walks_and_times_for_all_nodes(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_total,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);

                std::vector<std::vector<std::tuple<int, int64_t>>> result;
                result.reserve(walks_with_times.size());

                for (const auto& walk : walks_with_times) {
                    std::vector<std::tuple<int, int64_t>> converted_walk;
                    converted_walk.reserve(walk.size());

                    for (const auto& node_time : walk) {
                        converted_walk.emplace_back(node_time.node, node_time.timestamp);
                    }

                    result.push_back(std::move(converted_walk));
                }

                return result;
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
                num_walks_total (int): Total number of walks.
                initial_edge_bias (str, optional): Bias type for first edge selection.
                    Uses walk_bias if not specified.
                walk_direction (str, optional): Direction of temporal random walks.
                    Either "Forward_In_Time" (default) or "Backward_In_Time".

            Returns:
                List[List[Tuple[int, int]]]: List of walks as (node_id, timestamp) sequences.
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_walks_total"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks", [](TemporalRandomWalk& tw,
                                    const int max_walk_len,
                                    const std::string& walk_bias,
                                    const int num_walks_total,
                                    const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                    const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 return tw.get_random_walks(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_total,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);
             },
             R"(
            Generates temporal random walks.

            Args:
                max_walk_len (int): Maximum length of each random walk
                walk_bias (str): Type of bias for selecting next edges during walk.
                    Can be one of:
                        - "Uniform": Equal probability for all valid edges
                        - "Linear": Linear decay based on time
                        - "ExponentialIndex": Exponential decay with index sampling
                        - "ExponentialWeight": Exponential decay with timestamp weights
                num_walks_per_node (int): Number of walks to generate per node
                initial_edge_bias (str, optional): Bias type for selecting first edge.
                    Uses walk_bias if not specified.
                walk_direction (str, optional): Direction of temporal random walk.
                    Either "Forward_In_Time" (default) or "Backward_In_Time"

            Returns:
                List[List[int]]: A list of walks, where each walk is a list of node IDs
                    representing a temporal path through the network.
            )")

        .def("get_random_walks_and_times", [](TemporalRandomWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const int num_walks_per_node,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 auto walks_with_times = tw.get_random_walks_and_times(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_per_node,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);

                std::vector<std::vector<std::tuple<int, int64_t>>> result;
                result.reserve(walks_with_times.size());

                for (const auto& walk : walks_with_times) {
                    std::vector<std::tuple<int, int64_t>> converted_walk;
                    converted_walk.reserve(walk.size());

                    for (const auto& node_time : walk) {
                        converted_walk.emplace_back(node_time.node, node_time.timestamp);
                    }

                    result.push_back(std::move(converted_walk));
                }

                return result;
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
                num_walks_total (int): Total number of walks.
                initial_edge_bias (str, optional): Bias type for first edge selection.
                    Uses walk_bias if not specified.
                walk_direction (str, optional): Direction of temporal random walks.
                    Either "Forward_In_Time" (default) or "Backward_In_Time".

            Returns:
                List[List[Tuple[int, int]]]: List of walks where each walk is a sequence of
                    (node_id, timestamp) pairs representing temporal paths through the network.
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
        .def("to_networkx", [](const TemporalRandomWalk& tw)
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
