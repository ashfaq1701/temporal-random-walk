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
    NODE_GROUPED,
    // Ablation variant: same scheduler pipeline as NODE_GROUPED, but the
    // G-partition routes every cooperative task to the `*_global` kernel
    // tier — the per-node smem panel preload is bypassed. Isolates the
    // cooperation contribution from the smem-preload contribution in
    // paper-style ablations (FULL_WALK vs COOP_WITHOUT_SMEM vs
    // COOP_WITH_SMEM). Produces the same walk distribution as NODE_GROUPED.
    NODE_GROUPED_GLOBAL_ONLY
};

// Default kernel path for the public walk API. NODE_GROUPED is the
// cooperative per-step sampling pipeline; FULL_WALK (one thread per
// walk for the walk's whole life) is retained as a fallback and
// baseline. All default arguments that take KernelLaunchType should
// reference this constant rather than spelling out the enum value.
constexpr KernelLaunchType DEFAULT_KERNEL_LAUNCH_TYPE =
    KernelLaunchType::NODE_GROUPED;

#endif // ENUMS_H
