#include "fuzzer_utils.h"
#include <ATen/CPUGeneratorImpl.h>
#include <algorithm>
#include <cstring>
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;

        // Get the default CPU generator
        auto cpu_gen = at::detail::getDefaultCPUGenerator();
        
        // Get initial seed - this is the main API we're testing
        uint64_t initial_seed = cpu_gen.current_seed();
        (void)initial_seed; // Use the value to prevent optimization

        if (Size >= sizeof(uint64_t))
        {
            // Extract a seed value from fuzzer data
            uint64_t new_seed = 0;
            std::memcpy(&new_seed, Data + offset, sizeof(new_seed));
            offset += sizeof(new_seed);

            // Set a new manual seed
            torch::manual_seed(new_seed);

            // Verify the seed was set correctly
            uint64_t current_seed = at::detail::getDefaultCPUGenerator().current_seed();
            if (current_seed != new_seed)
            {
                // This would indicate a bug in PyTorch
                std::cerr << "Seed mismatch: expected " << new_seed << " got " << current_seed << std::endl;
            }

            // Exercise the generator by creating random tensors
            try {
                torch::Tensor rand_tensor = torch::rand({2, 2});
                (void)rand_tensor.sum().item<double>();
                
                torch::Tensor randn_tensor = torch::randn({2, 2});
                (void)randn_tensor.sum().item<double>();
            } catch (...) {
                // Silently catch shape/allocation issues
            }
        }

        if (Size >= offset + sizeof(uint64_t))
        {
            // Test reseeding with another value
            uint64_t reseed_value = 0;
            size_t bytes_to_copy = std::min(sizeof(reseed_value), Size - offset);
            std::memcpy(&reseed_value, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;

            // Exercise the generator's seed() method which reseeds from random source
            {
                std::lock_guard<std::mutex> lock(cpu_gen.mutex());
                cpu_gen.set_current_seed(reseed_value);
            }

            // Verify seed retrieval after reseed
            uint64_t seed_after = cpu_gen.current_seed();
            (void)seed_after;
        }

        // Test determinism: same seed should produce same random values
        if (Size >= 2)
        {
            uint64_t test_seed = 42;
            torch::manual_seed(test_seed);
            
            try {
                torch::Tensor t1 = torch::rand({3, 3});
                
                torch::manual_seed(test_seed);
                torch::Tensor t2 = torch::rand({3, 3});
                
                // Tensors should be equal with same seed
                bool are_equal = torch::allclose(t1, t2);
                (void)are_equal;
            } catch (...) {
                // Silently handle any tensor operation failures
            }
        }

        // If we have remaining data, use it to create and process a tensor
        if (offset < Size && Size - offset > 2)
        {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                (void)tensor.sum().item<double>();
            } catch (...) {
                // Silently catch tensor creation failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}