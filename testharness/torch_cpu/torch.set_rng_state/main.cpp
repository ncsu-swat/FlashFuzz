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
        
        // Need at least some data to create a tensor for RNG state
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to use as RNG state
        torch::Tensor state_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // RNG state should be a 1D ByteTensor
        // Try to make it compatible with set_rng_state requirements
        if (state_tensor.dim() != 1 || state_tensor.dtype() != torch::kUInt8) {
            // Convert to 1D ByteTensor if needed
            if (state_tensor.numel() > 0) {
                // Reshape to 1D and convert to uint8
                state_tensor = state_tensor.reshape({-1}).to(torch::kUInt8);
            } else {
                // Create a minimal valid state tensor if empty
                state_tensor = torch::ones({5}, torch::kUInt8);
            }
        }
        
        // Get current RNG state to restore later
        auto original_state = torch::default_generator().get_state();
        
        // Apply the operation - set the RNG state
        torch::default_generator().set_state(state_tensor);
        
        // Test that the RNG state was set by generating a random number
        auto random_tensor = torch::rand({2, 2});
        
        // Restore original RNG state
        torch::default_generator().set_state(original_state);
        
        // Try with edge cases if we have more data
        if (offset + 1 < Size) {
            // Try with empty tensor
            torch::Tensor empty_tensor = torch::empty({0}, torch::kUInt8);
            try {
                torch::default_generator().set_state(empty_tensor);
            } catch (...) {
                // Expected to fail, continue
            }
            
            // Try with very large tensor
            if (offset + 2 < Size) {
                uint8_t size_factor = Data[offset++];
                int64_t large_size = 1000 + (size_factor % 10000);
                torch::Tensor large_tensor = torch::ones({large_size}, torch::kUInt8);
                try {
                    torch::default_generator().set_state(large_tensor);
                } catch (...) {
                    // May fail, continue
                }
            }
            
            // Try with different data types
            std::vector<torch::ScalarType> dtypes = {
                torch::kFloat, torch::kDouble, torch::kInt32, torch::kInt64, 
                torch::kBool, torch::kComplexFloat
            };
            
            uint8_t dtype_idx = Data[offset++] % dtypes.size();
            torch::Tensor wrong_dtype = torch::ones({5}, dtypes[dtype_idx]);
            try {
                torch::default_generator().set_state(wrong_dtype);
            } catch (...) {
                // Expected to fail for non-uint8, continue
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
