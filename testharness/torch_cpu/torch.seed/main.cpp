#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Test torch::seed() - generates and sets a random seed, returns it
        uint64_t generated_seed = torch::seed();
        
        // Create a tensor using the current random state
        auto random_tensor = torch::rand({3, 3});
        
        // Test calling seed() again
        uint64_t another_generated_seed = torch::seed();
        
        // Create another tensor with the new random state
        auto random_tensor2 = torch::rand({3, 3});
        
        // Also test manual_seed for comparison - use fuzzer data if available
        if (Size >= sizeof(int64_t)) {
            int64_t seed;
            std::memcpy(&seed, Data, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::manual_seed(seed);
            auto random_tensor3 = torch::rand({3, 3});
            
            // Re-set with torch::seed() to verify it works after manual_seed
            uint64_t new_seed = torch::seed();
            auto random_tensor4 = torch::rand({3, 3});
        }
        
        // Test multiple calls to seed() in sequence
        for (int i = 0; i < 3; i++) {
            uint64_t s = torch::seed();
            auto t = torch::rand({2, 2});
        }
        
        // Test with extreme seed value via manual_seed if we have more data
        if (Size >= offset + sizeof(int64_t)) {
            int64_t extreme_seed;
            std::memcpy(&extreme_seed, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::manual_seed(extreme_seed);
            auto extreme_tensor = torch::rand({2, 2});
        }
        
        // Final call to seed() to restore random seeding
        uint64_t final_seed = torch::seed();
        auto final_tensor = torch::rand({2, 2});
        
        // Suppress unused variable warnings
        (void)generated_seed;
        (void)another_generated_seed;
        (void)final_seed;
        (void)random_tensor;
        (void)random_tensor2;
        (void)final_tensor;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}