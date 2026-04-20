#ifndef PICKER_DISPATCH_CUH
#define PICKER_DISPATCH_CUH

#include "../data/enums.cuh"

// Dispatch a runtime RandomPickerType / bool value to a generic-lambda
// visitor that pulls the value out as a constexpr template parameter
// via tag dispatch. C++17-compatible (no template-parameter-list on
// generic lambdas required).
//
// Usage:
//   dispatch_picker_type(pt, [&](auto tag) {
//       constexpr auto value = decltype(tag)::value;
//       // value is a compile-time RandomPickerType
//   });

template <RandomPickerType PT>
struct PickerTag {
    static constexpr RandomPickerType value = PT;
};

template <bool B>
struct BoolTag {
    static constexpr bool value = B;
};

template <typename Visitor>
inline void dispatch_picker_type(RandomPickerType pt, Visitor&& visit) {
    switch (pt) {
        case RandomPickerType::Uniform:
            visit(PickerTag<RandomPickerType::Uniform>{}); break;
        case RandomPickerType::Linear:
            visit(PickerTag<RandomPickerType::Linear>{}); break;
        case RandomPickerType::ExponentialIndex:
            visit(PickerTag<RandomPickerType::ExponentialIndex>{}); break;
        case RandomPickerType::ExponentialWeight:
            visit(PickerTag<RandomPickerType::ExponentialWeight>{}); break;
        case RandomPickerType::TemporalNode2Vec:
            visit(PickerTag<RandomPickerType::TemporalNode2Vec>{}); break;
        case RandomPickerType::TEST_FIRST:
            visit(PickerTag<RandomPickerType::TEST_FIRST>{}); break;
        case RandomPickerType::TEST_LAST:
            visit(PickerTag<RandomPickerType::TEST_LAST>{}); break;
    }
}

template <typename Visitor>
inline void dispatch_bool(bool b, Visitor&& visit) {
    if (b) visit(BoolTag<true>{});
    else   visit(BoolTag<false>{});
}

#endif // PICKER_DISPATCH_CUH
