#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Get the current RNG state
        auto rng_state = torch::default_generator->get_state();
        
        // If we have enough data, try to set a new RNG state
        if (Size > 0) {
            // Create a tensor from the input data
            torch::Tensor input_tensor;
            
            try {
                input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try to set the RNG state with the input tensor
                // This might throw if the tensor is not a valid RNG state
                if (input_tensor.dtype() == torch::kByte) {
                    torch::default_generator->set_state(input_tensor);
                    
                    // Get the RNG state again after setting it
                    auto new_rng_state = torch::default_generator->get_state();
                    
                    // Generate some random numbers to test the RNG
                    auto random_tensor = torch::rand({3, 3});
                    
                    // Restore the original RNG state
                    torch::default_generator->set_state(rng_state);
                }
            } catch (...) {
                // If setting the RNG state fails, that's expected for invalid inputs
                // Just continue with the test
            }
        }
        
        // Test get_rng_state with different device options
        try {
            // Try to get RNG state for CPU
            auto cpu_rng_state = torch::default_generator->get_state();
            
            // Try to get RNG state for CUDA if available
            if (torch::cuda::is_available()) {
                for (int device_idx = 0; device_idx < torch::cuda::device_count(); ++device_idx) {
                    auto cuda_device = torch::Device(torch::kCUDA, device_idx);
                    auto cuda_generator = torch::cuda::detail::getDefaultCUDAGenerator(device_idx);
                    auto cuda_rng_state = cuda_generator.get_state();
                }
            }
        } catch (...) {
            // If getting RNG state for a device fails, that's useful information
            // but we should continue with the test
        }
        
        // Test manual seed functions which interact with RNG state
        if (Size >= sizeof(int64_t)) {
            int64_t seed_value;
            std::memcpy(&seed_value, Data, sizeof(int64_t));
            
            // Set a manual seed
            torch::manual_seed(seed_value);
            
            // Get the RNG state after setting the seed
            auto seeded_rng_state = torch::default_generator->get_state();
            
            // Try CUDA manual seed if available
            if (torch::cuda::is_available()) {
                torch::cuda::manual_seed(seed_value);
                torch::cuda::manual_seed_all(seed_value);
            }
        }
        
        // Test seed function which relates to RNG state
        auto current_seed = torch::default_generator->current_seed();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}