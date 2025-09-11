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
        
        // Get the initial seed value
        uint64_t initial_seed = torch::default_generator->current_seed();
        
        // Try to create a tensor if we have enough data
        if (Size > 2) {
            // Create a tensor using the fuzzer data
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Get the initial seed again after tensor creation
            uint64_t seed_after_tensor = torch::default_generator->current_seed();
            
            // Check if the seed changed (it shouldn't)
            if (initial_seed != seed_after_tensor) {
                throw std::runtime_error("Initial seed changed unexpectedly");
            }
            
            // Try to manually set a seed if we have more data
            if (offset < Size) {
                uint64_t new_seed = 0;
                size_t bytes_to_copy = std::min(sizeof(new_seed), Size - offset);
                std::memcpy(&new_seed, Data + offset, bytes_to_copy);
                
                // Set the seed
                torch::manual_seed(new_seed);
                
                // Verify the seed was set correctly
                uint64_t current_seed = torch::default_generator->current_seed();
                
                // Check if the seed was set correctly
                if (current_seed != new_seed) {
                    throw std::runtime_error("Seed was not set correctly");
                }
            }
        }
        
        // Test with different device types if available
        if (torch::cuda::is_available()) {
            uint64_t cuda_seed = torch::cuda::default_generator(0)->current_seed();
            
            // Try to set a CUDA seed if we have more data
            if (offset < Size) {
                uint64_t new_cuda_seed = 0;
                size_t bytes_to_copy = std::min(sizeof(new_cuda_seed), Size - offset);
                std::memcpy(&new_cuda_seed, Data + offset, bytes_to_copy);
                
                // Set the CUDA seed
                torch::cuda::manual_seed(new_cuda_seed);
                
                // Verify the CUDA seed was set correctly
                uint64_t current_cuda_seed = torch::cuda::default_generator(0)->current_seed();
                
                // Check if the CUDA seed was set correctly
                if (current_cuda_seed != new_cuda_seed) {
                    throw std::runtime_error("CUDA seed was not set correctly");
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
