#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include "temporal_random_walk_proxy.h"
#include "random_picker_proxies.h"
#include <stdexcept>
#include "../src/structs/enums.h"
#include "../src/structs/structs.cuh"
#include "../src/core/TemporalRandomWalk.cuh"


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

GPUUsageMode gpu_usage_mode_from_string(const std::string& gpu_usage_mode_str)
{
    if (gpu_usage_mode_str == "ON_CPU")
    {
        return GPUUsageMode::ON_CPU;
    }
    else if (gpu_usage_mode_str == "ON_GPU")
    {
        return GPUUsageMode::ON_GPU;
    }
    else
    {
        throw std::invalid_argument("Invalid usage mode: " + gpu_usage_mode_str);
    }
}

PYBIND11_MODULE(_temporal_random_walk, m)
{
    py::class_<TemporalRandomWalkProxy>(m, "TemporalRandomWalk")
        .def(py::init([](const bool is_directed, const std::optional<std::string>& gpu_usage_mode,
                         const std::optional<int64_t> max_time_capacity, std::optional<bool> enable_weight_computation,
                         std::optional<double> timescale_bound)
             {
                 return std::make_unique<TemporalRandomWalkProxy>(
                     is_directed,
                     gpu_usage_mode_from_string(gpu_usage_mode.value_or("ON_CPU")),
                     max_time_capacity.value_or(-1),
                     enable_weight_computation.value_or(false),
                     timescale_bound.value_or(DEFAULT_TIMESCALE_BOUND));
             }),
             R"(
            Initialize a temporal random walk generator.

            Args:
            is_directed (bool): Whether to create a directed graph.
            gpu_usage_mode (str, optional): GPU usage mode ("ON_CPU", "ON_GPU"). Default: "ON_CPU".
            max_time_capacity (int, optional): Maximum time window for edges. Edges older than (latest_time - max_time_capacity) are removed. Use -1 for no limit. Defaults to -1.
            enable_weight_computation (bool, optional): Enable CTDNE weight computation. Required for ExponentialWeight picker. Defaults to False.
            timescale_bound (float, optional): Scale factor for temporal differences. Used to prevent numerical issues with large time differences. Defaults to 50.0.
            )",
             py::arg("is_directed"),
             py::arg("gpu_usage_mode") = "USE_CPU",
             py::arg("max_time_capacity") = py::none(),
             py::arg("enable_weight_computation") = py::none(),
             py::arg("timescale_bound") = py::none())
        .def("add_multiple_edges", [](TemporalRandomWalkProxy& tw, const std::vector<std::tuple<int, int, int64_t>>& edge_infos)
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
        .def("get_random_walks_for_all_nodes", [](TemporalRandomWalkProxy& tw,
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

        .def("get_random_walks_and_times_for_all_nodes", [](TemporalRandomWalkProxy& tw,
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

                 const auto walks_with_times = tw.get_random_walks_and_times_for_all_nodes(
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
                List[List[Tuple[int, int]]]: List of walks as (node_id, timestamp) sequences.
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_walks_per_node"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks", [](TemporalRandomWalkProxy& tw,
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

                 return tw.get_random_walks(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_per_node,
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

        .def("get_random_walks_and_times", [](TemporalRandomWalkProxy& tw,
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
                num_walks_per_node (int): Number of walks per starting node.
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
             py::arg("num_walks_per_node"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks_with_specific_number_of_contexts", [](TemporalRandomWalkProxy& tw,
                                    const int max_walk_len,
                                    const std::string& walk_bias,
                                    const std::optional<long> num_cw = std::nullopt,
                                    const std::optional<int> num_walks_per_node = std::nullopt,
                                    const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                    const std::string& walk_direction = "Forward_In_Time",
                                    const std::optional<int> context_window_len = std::nullopt,
                                    const float p_walk_success_threshold = DEFAULT_SUCCESS_THRESHOLD)
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 return tw.get_random_walks_with_specific_number_of_contexts(
                     max_walk_len,
                     &walk_bias_enum,
                     num_cw.value_or(-1),
                     num_walks_per_node.value_or(-1),
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum,
                     context_window_len.value_or(-1),
                     p_walk_success_threshold);
             },
             R"(
            Generate temporal random walks with a specific number of contexts.

            The number of walks can vary based on their actual lengths as this method ensures
            a fixed number of context windows rather than a fixed number of walks.

            Args:
                max_walk_len (int): Maximum length of each random walk.
                walk_bias (str): Type of bias for selecting next node.
                    Choices:
                        - "Uniform": Equal probability for all edges
                        - "Linear": Linear time decay
                        - "ExponentialIndex": Exponential decay with indices
                        - "ExponentialWeight": Exponential decay with weights
                num_cw (int, optional): Target number of context windows to generate.
                    If not specified, calculated using num_walks_per_node.
                num_walks_per_node (int, optional): Number of walks per starting node.
                    Only used if num_cw is not specified.
                initial_edge_bias (str, optional): Bias type for first edge selection.
                    Uses walk_bias if not specified.
                walk_direction (str, optional): Direction of temporal random walks.
                    Either "Forward_In_Time" (default) or "Backward_In_Time".
                context_window_len (int, optional): Minimum length of each walk.
                    Defaults to 2 if not specified.
                p_walk_success_threshold (float, optional): Minimum required success rate
                    for walk generation. Default: 0.01

                Returns:
                    List[List[int]]: List of walks where each walk is a sequence of node IDs
                        representing temporal paths through the network.
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_cw") = py::none(),
             py::arg("num_walks_per_node") = py::none(),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time",
             py::arg("context_window_len") = py::none(),
             py::arg("p_walk_success_threshold") = DEFAULT_SUCCESS_THRESHOLD)

        .def("get_random_walks_and_times_with_specific_number_of_contexts", [](TemporalRandomWalkProxy& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const std::optional<long> num_cw = std::nullopt,
                                               const std::optional<int> num_walks_per_node = std::nullopt,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time",
                                               const std::optional<int> context_window_len = std::nullopt,
                                               const float p_walk_success_threshold = DEFAULT_SUCCESS_THRESHOLD)
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 auto walks_with_times = tw.get_random_walks_and_times_with_specific_number_of_contexts(
                     max_walk_len,
                     &walk_bias_enum,
                     num_cw.value_or(-1),
                     num_walks_per_node.value_or(-1),
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum,
                     context_window_len.value_or(-1),
                     p_walk_success_threshold);

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
             Generate temporal random walks with timestamps and specific number of contexts.

            The number of walks can vary based on their actual lengths as this method ensura fixed number of context windows rather than a fixed number of walks.

            Args:
                max_walk_len (int): Maximum length of each random walk.
                walk_bias (str): Type of bias for selecting next node.
                    Choices:
                        - "Uniform": Equal probability for all edges
                        - "Linear": Linear time decay
                        - "ExponentialIndex": Exponential decay with indices
                        - "ExponentialWeight": Exponential decay with weights
                num_cw (int, optional): Target number of context windows to generate.
                    If not specified, calculated using num_walks_per_node.
                num_walks_per_node (int, optional): Number of walks per starting node.
                    Only used if num_cw is not specified.
                initial_edge_bias (str, optional): Bias type for first edge selection.
                    Uses walk_bias if not specified.
                walk_direction (str, optional): Direction of temporal random walks.
                    Either "Forward_In_Time" (default) or "Backward_In_Time".
                context_window_len (int, optional): Minimum length of each walk.
                    Defaults to 2 if not specified.
                p_walk_success_threshold (float, optional): Minimum required success rate
                    for walk generation. Default: 0.01

            Returns:
                List[List[Tuple[int, int]]]: List of walks where each walk is a sequence of
                    (node_id, timestamp) pairs representing temporal paths through the network.
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_cw") = py::none(),
             py::arg("num_walks_per_node") = py::none(),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time",
             py::arg("context_window_len") = py::none(),
             py::arg("p_walk_success_threshold") = DEFAULT_SUCCESS_THRESHOLD)

        .def("get_node_count", &TemporalRandomWalkProxy::get_node_count,
            R"(
            Get total number of nodes in the graph.

            Returns:
                int: Number of active nodes.
            )")
        .def("get_edge_count", &TemporalRandomWalkProxy::get_edge_count,
             R"(
             Returns the total number of directed edges in the temporal graph.

             Returns:
                int: The total number of directed edges.
             )")
        .def("get_node_ids", [](const TemporalRandomWalkProxy& tw)
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
        .def("clear", &TemporalRandomWalkProxy::clear,
             R"(
            Clears and reinitiates the underlying graph.
            )"
        )
        .def("add_edges_from_networkx", [](TemporalRandomWalkProxy& tw, const py::object& nx_graph)
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
        .def("to_networkx", [](const TemporalRandomWalkProxy& tw)
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

    py::class_<LinearRandomPickerProxy>(m, "LinearRandomPicker")
        .def(py::init([](const std::optional<std::string>& gpu_usage_mode)
             {
                 return LinearRandomPickerProxy(
                     gpu_usage_mode_from_string(gpu_usage_mode.value_or("ON_CPU")));
             }),
             R"(
            Initialize linear time decay random picker.

            Args:
                gpu_usage_mode (str, optional): GPU usage mode ("ON_CPU", "ON_GPU"). Default: "ON_CPU"
            )",
             py::arg("gpu_usage_mode") = "ON_CPU")

        .def("pick_random", &LinearRandomPickerProxy::pick_random,
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

    py::class_<ExponentialIndexRandomPickerProxy>(m, "ExponentialIndexRandomPicker")
        .def(py::init([](const std::optional<std::string>& gpu_usage_mode)
             {
                 return ExponentialIndexRandomPickerProxy(
                     gpu_usage_mode_from_string(gpu_usage_mode.value_or("ON_CPU")));
             }),
             R"(
            Initialize index based exponential time decay random picker.

            Args:
                gpu_usage_mode (str, optional): GPU usage mode ("ON_CPU", "ON_GPU"). Default: "ON_CPU"
            )",
             py::arg("gpu_usage_mode") = "ON_CPU")

        .def("pick_random", &ExponentialIndexRandomPickerProxy::pick_random,
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

    py::class_<UniformRandomPickerProxy>(m, "UniformRandomPicker")
        .def(py::init([](const std::optional<std::string>& gpu_usage_mode)
             {
                 return UniformRandomPickerProxy(
                     gpu_usage_mode_from_string(gpu_usage_mode.value_or("ON_CPU")));
             }),
             R"(
            Initialize uniform random picker.

            Args:
                gpu_usage_mode (str, optional): GPU usage mode ("ON_CPU", "ON_GPU"). Default: "ON_CPU"
            )",
             py::arg("gpu_usage_mode") = "ON_CPU")

        .def("pick_random", &UniformRandomPickerProxy::pick_random,
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

    py::class_<WeightBasedRandomPickerProxy>(m, "WeightBasedRandomPicker")
        .def(
            py::init([](){ return WeightBasedRandomPickerProxy(GPUUsageMode::ON_CPU); }),
            R"(
            Initialize exponential time decay random picker with weight-based sampling.

            For use with CTDNE temporal random walks where edge selection probabilities are weighted
            by temporal differences.
            )")
        .def("pick_random", py::overload_cast<const std::vector<double>&, int, int>(&WeightBasedRandomPickerProxy::pick_random),
            R"(
            Pick random index based on cumulative temporal weights.

            Args:
                cumulative_weights (List[float]): Array of cumulative weights for sampling.
                    Must be monotonically increasing.
                group_start (int): Start index of the group (inclusive)
                group_end (int): End index of the group (exclusive)

            Returns:
                int: Selected index based on the weight distribution
            )",
            py::arg("cumulative_weights"), py::arg("group_start"), py::arg("group_end"));
}
