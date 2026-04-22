#ifndef ENUMS_H
#define ENUMS_H

enum RandomPickerType {
    Uniform,
    Linear,
    ExponentialIndex,
    ExponentialWeight,
    TemporalNode2Vec,

    // ONLY FOR TESTS
    TEST_FIRST,
    TEST_LAST
};

enum WalkDirection {
    Forward_In_Time,
    Backward_In_Time
};

enum KernelLaunchType {
    FULL_WALK,
    NODE_GROUPED
};

// Default kernel path for the public walk API. NODE_GROUPED is the
// cooperative per-step sampling pipeline; FULL_WALK (one thread per
// walk for the walk's whole life) is retained as a fallback and
// baseline. All default arguments that take KernelLaunchType should
// reference this constant rather than spelling out the enum value.
constexpr KernelLaunchType DEFAULT_KERNEL_LAUNCH_TYPE =
    KernelLaunchType::NODE_GROUPED;

#endif // ENUMS_H
