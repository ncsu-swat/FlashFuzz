#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Extract a seed value from the input data if available
        int64_t seed = 0;
        if (Size >= sizeof(int64_t)) {
            std::memcpy(&seed, Data, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Test torch::manual_seed with the extracted seed
        torch::manual_seed(seed);
        
        // Create a tensor using the current random state
        auto random_tensor = torch::rand({3, 3});
        
        // Test setting seed again
        torch::manual_seed(seed);
        
        // Create another tensor with the same seed to verify determinism
        auto random_tensor2 = torch::rand({3, 3});
        
        // Test with a different seed if we have more data
        if (Size >= offset + sizeof(int64_t)) {
            int64_t another_seed;
            std::memcpy(&another_seed, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::manual_seed(another_seed);
            auto random_tensor3 = torch::rand({3, 3});
        }
        
        // Test with a potentially extreme seed value
        if (Size >= offset + sizeof(int64_t)) {
            int64_t extreme_seed;
            std::memcpy(&extreme_seed, Data + offset, sizeof(int64_t));
            torch::manual_seed(extreme_seed);
        }
        
        // Create final tensor after all seed operations
        auto final_tensor = torch::rand({2, 2});
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
