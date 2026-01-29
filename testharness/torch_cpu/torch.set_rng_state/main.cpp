#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <ATen/CPUGeneratorImpl.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;

        // Get the default CPU generator
        auto& generator = at::detail::getDefaultCPUGenerator();
        
        // Get current RNG state to understand valid format and to restore later
        torch::Tensor original_state = generator.get_state();
        
        // Test 1: Save and restore original state (basic functionality)
        generator.set_state(original_state);
        
        // Generate random tensor to verify RNG works
        auto random1 = torch::rand({2, 2});
        
        // Test 2: Modify state based on fuzzer data and set it back
        // Clone the original state and modify some bytes based on fuzzer input
        torch::Tensor modified_state = original_state.clone();
        
        // Modify state bytes using fuzzer data (safely within bounds)
        int64_t state_size = modified_state.numel();
        if (state_size > 0) {
            auto state_data = modified_state.data_ptr<uint8_t>();
            
            // Use fuzzer data to modify parts of the state
            size_t num_modifications = std::min(Size - offset, static_cast<size_t>(state_size));
            for (size_t i = 0; i < num_modifications && offset < Size; i++) {
                int64_t idx = Data[offset++] % state_size;
                if (offset < Size) {
                    state_data[idx] = Data[offset++];
                }
            }
            
            // Try to set the modified state - may fail if state becomes invalid
            try {
                generator.set_state(modified_state);
                // Generate random to exercise the modified state
                auto random2 = torch::rand({3, 3});
            } catch (...) {
                // Modified state may be invalid, silently continue
            }
        }
        
        // Restore original state
        generator.set_state(original_state);
        
        // Test 3: Test with invalid inputs (expected to fail)
        if (offset < Size) {
            // Try with empty tensor
            torch::Tensor empty_tensor = torch::empty({0}, torch::kByte);
            try {
                generator.set_state(empty_tensor);
            } catch (...) {
                // Expected to fail
            }
            
            // Try with wrong size tensor
            torch::Tensor wrong_size = torch::ones({10}, torch::kByte);
            try {
                generator.set_state(wrong_size);
            } catch (...) {
                // Expected to fail
            }
            
            // Try with wrong dtype
            torch::Tensor wrong_dtype = torch::ones({state_size}, torch::kFloat32);
            try {
                generator.set_state(wrong_dtype);
            } catch (...) {
                // Expected to fail
            }
        }
        
        // Always restore original state at the end
        generator.set_state(original_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}