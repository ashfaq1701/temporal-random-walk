#include <iostream>
#include <proxies/RandomPicker.cuh>

constexpr int TOTAL_TIMESTEPS = 100000000;

#ifdef HAS_CUDA
constexpr bool USE_GPU = true;
#else
constexpr bool USE_GPU = false;
#endif

int main()
{
    ExponentialIndexRandomPicker random_picker(USE_GPU);

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
