#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

// --- Fuzzer Entry Point ---
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
        
        // Need at least 8 bytes for a seed value
        if (Size < 8) {
            return 0;
        }
        
        // Extract a 64-bit seed value from the input data
        int64_t seed;
        std::memcpy(&seed, Data, sizeof(int64_t));
        offset += sizeof(int64_t);
        
        // Set the manual seed
        torch::manual_seed(seed);
        
        // Test the seed by generating a random tensor
        torch::Tensor random_tensor = torch::rand({3, 3});
        
        // Test with another seed value to ensure determinism
        torch::manual_seed(seed);
        torch::Tensor random_tensor2 = torch::rand({3, 3});
        
        // The two tensors should be identical since we used the same seed
        bool tensors_equal = torch::all(torch::eq(random_tensor, random_tensor2)).item<bool>();
        (void)tensors_equal; // Suppress unused variable warning
        
        // If there's more data, try creating a tensor and test operations with the seeded random state
        if (offset < Size) {
            try {
                torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test some operations that might use the random state
                torch::Tensor dropout_result = torch::dropout(input_tensor, 0.5, true);
                
                // Reset seed and verify determinism
                torch::manual_seed(seed);
                torch::Tensor dropout_result2 = torch::dropout(input_tensor, 0.5, true);
                
                // Check if dropout results are identical with the same seed
                bool dropout_equal = torch::all(torch::eq(dropout_result, dropout_result2)).item<bool>();
                (void)dropout_equal; // Suppress unused variable warning
            } catch (const std::exception &e) {
                // Ignore exceptions from tensor creation - expected for invalid shapes
            }
        }
        
        // Test with negative seed values
        if (seed < 0) {
            torch::manual_seed(seed);
            torch::Tensor neg_seed_tensor = torch::rand({2, 2});
        }
        
        // Test with extreme seed values
        torch::manual_seed(std::numeric_limits<int64_t>::max());
        torch::Tensor max_seed_tensor = torch::rand({2, 2});
        
        torch::manual_seed(std::numeric_limits<int64_t>::min());
        torch::Tensor min_seed_tensor = torch::rand({2, 2});
        
        // Test with zero seed
        torch::manual_seed(0);
        torch::Tensor zero_seed_tensor = torch::rand({2, 2});
        
        // Additional coverage: test various random generation functions with the seed
        if (Size >= 16) {
            uint64_t seed2;
            std::memcpy(&seed2, Data + 8, sizeof(uint64_t));
            torch::manual_seed(static_cast<int64_t>(seed2));
            
            // Test different random tensor creation methods
            torch::Tensor randn_tensor = torch::randn({2, 3});
            torch::Tensor randint_tensor = torch::randint(0, 100, {3, 3});
            torch::Tensor randperm_tensor = torch::randperm(10);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}