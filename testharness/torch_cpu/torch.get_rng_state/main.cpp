#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include <algorithm> // For std::min
#include <cstring>  // For std::memcpy
#include <iostream> // For cerr

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
        
        // Get the default CPU generator
        at::Generator cpu_gen = at::detail::getDefaultCPUGenerator();
        
        // Get the current RNG state (this is the main API we're testing)
        auto rng_state = cpu_gen.get_state();
        
        // Verify we got a valid state tensor
        (void)rng_state.numel();
        (void)rng_state.dtype();
        
        // If we have enough data, try to set a new RNG state and then get it back
        if (Size > 0) {
            try {
                torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try to set the RNG state with the input tensor
                // This might throw if the tensor is not a valid RNG state
                if (input_tensor.dtype() == torch::kByte && input_tensor.is_contiguous() && input_tensor.dim() == 1) {
                    auto gen_copy = cpu_gen.clone();
                    gen_copy.set_state(input_tensor);
                    
                    // Get the RNG state again after setting it - main API test
                    auto new_rng_state = gen_copy.get_state();
                    (void)new_rng_state.numel();
                    
                    // Generate some random numbers to verify RNG works
                    auto random_tensor = torch::rand({3, 3});
                    (void)random_tensor.sum().item<double>();
                    
                    // Restore the original RNG state on the clone and verify get_state works
                    gen_copy.set_state(rng_state);
                    auto restored_state = gen_copy.get_state();
                    (void)restored_state.numel();
                }
            } catch (...) {
                // If setting the RNG state fails, that's expected for invalid inputs
            }
        }
        
        // Test get_rng_state with cloned generators
        try {
            auto cloned_gen = cpu_gen.clone();
            auto cloned_state = cloned_gen.get_state();
            (void)cloned_state.numel();
            (void)cloned_state.size(0);
        } catch (...) {
            // Cloning or getting state might fail in edge cases
        }
        
#ifdef USE_CUDA
        // Test get_rng_state for CUDA if available
        try {
            if (torch::cuda::is_available()) {
                const int device_count = std::min<int>(torch::cuda::device_count(), 2);
                for (int device_idx = 0; device_idx < device_count; ++device_idx) {
                    auto cuda_generator = at::cuda::detail::getDefaultCUDAGenerator(device_idx);
                    auto cuda_rng_state = cuda_generator.get_state();
                    (void)cuda_rng_state.numel();
                }
            }
        } catch (...) {
            // CUDA operations might fail
        }
#endif
        
        // Test manual seed functions which interact with RNG state
        if (Size >= sizeof(int64_t)) {
            int64_t seed_value;
            std::memcpy(&seed_value, Data, sizeof(int64_t));
            
            // Set a manual seed
            torch::manual_seed(seed_value);
            
            // Get the RNG state after setting the seed - verifies get_state works after seeding
            auto seeded_rng_state = cpu_gen.get_state();
            (void)seeded_rng_state.numel();
            
#ifdef USE_CUDA
            try {
                if (torch::cuda::is_available()) {
                    torch::cuda::manual_seed(seed_value);
                    torch::cuda::manual_seed_all(seed_value);
                }
            } catch (...) {
                // CUDA seed operations might fail
            }
#endif
        }
        
        // Test current_seed which relates to RNG state
        auto current_seed = cpu_gen.current_seed();
        (void)current_seed;
        
        // Additional coverage: get state multiple times to test consistency
        auto state1 = cpu_gen.get_state();
        auto state2 = cpu_gen.get_state();
        (void)state1.numel();
        (void)state2.numel();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}