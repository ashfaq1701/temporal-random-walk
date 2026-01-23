#ifndef ENUMS_H
#define ENUMS_H

enum RandomPickerType {
    Uniform,
    Linear,
    ExponentialIndex,
    ExponentialWeight,

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
    STEP_BASED
};

#endif // ENUMS_H
