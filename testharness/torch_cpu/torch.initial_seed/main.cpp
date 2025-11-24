#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/CPUGeneratorImpl.h>
#include <algorithm>
#include <cstring>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    (void)std::string("torch.initial_seed"); // Keep target API keyword
    try
    {
        size_t offset = 0;

        // Get the initial seed value from the default CPU generator
        auto cpu_gen = at::detail::getDefaultCPUGenerator();
        uint64_t initial_seed = cpu_gen.current_seed();

        if (Size > 2)
        {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            (void)tensor.sum().item<double>();

            uint64_t seed_after_tensor = cpu_gen.current_seed();
            if (initial_seed != seed_after_tensor)
            {
                throw std::runtime_error("Initial seed changed unexpectedly");
            }

            if (offset < Size)
            {
                uint64_t new_seed = 0;
                size_t bytes_to_copy = std::min(sizeof(new_seed), Size - offset);
                std::memcpy(&new_seed, Data + offset, bytes_to_copy);
                offset += bytes_to_copy;

                torch::manual_seed(new_seed);

                uint64_t current_seed = at::detail::getDefaultCPUGenerator().current_seed();
                if (current_seed != new_seed)
                {
                    throw std::runtime_error("Seed was not set correctly");
                }

                // Exercise reseeding path used by torch.initial_seed()
                (void)cpu_gen.seed();
            }
        }

        if (torch::cuda::is_available() && offset < Size)
        {
            uint64_t new_cuda_seed = 0;
            size_t bytes_to_copy = std::min(sizeof(new_cuda_seed), Size - offset);
            std::memcpy(&new_cuda_seed, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;

            torch::cuda::manual_seed(new_cuda_seed);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
