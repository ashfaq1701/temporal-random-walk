#include <iostream>
#include <proxies/RandomPickerProxies.cuh>

#include "../src/proxies/NodeEdgeIndexProxy.cuh"

constexpr int TOTAL_TIMESTEPS = 100000000;
constexpr bool USE_GPU = false;

int main()
{
    ExponentialIndexRandomPickerProxy random_picker(USE_GPU);

    std::cout << "Prioritizing end: " << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        std::cout << random_picker.pick_random(0, TOTAL_TIMESTEPS, true) << std::endl;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Prioritizing start: " << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        std::cout << random_picker.pick_random(0, TOTAL_TIMESTEPS, false) << std::endl;
    }

    return 0;
}
