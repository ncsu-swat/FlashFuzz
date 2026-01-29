#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        int64_t ndim = input.dim();
        
        // If tensor has no dimensions, skip
        if (ndim == 0) {
            return 0;
        }
        
        // Extract parameters for Flatten module from the remaining data
        int64_t start_dim = 1;  // Default value
        int64_t end_dim = -1;   // Default value
        
        // If we have more data, use it for start_dim (bounded to valid range)
        if (offset < Size) {
            uint8_t raw_byte = Data[offset++];
            // Map to range [-ndim, ndim-1] which are valid dimension indices
            start_dim = static_cast<int64_t>(raw_byte % (2 * ndim + 1)) - ndim;
        }
        
        // If we have more data, use it for end_dim (bounded to valid range)
        if (offset < Size) {
            uint8_t raw_byte = Data[offset++];
            // Map to range [-ndim, ndim-1] which are valid dimension indices
            end_dim = static_cast<int64_t>(raw_byte % (2 * ndim + 1)) - ndim;
        }
        
        // Test 1: Create and apply the Flatten module with FlattenOptions
        try {
            torch::nn::Flatten flatten_module(
                torch::nn::FlattenOptions().start_dim(start_dim).end_dim(end_dim)
            );
            torch::Tensor output = flatten_module->forward(input);
            
            // Ensure the output is computed (force evaluation)
            output.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for invalid dimension combinations
        }
        
        // Test 2: Use the functional API
        try {
            torch::Tensor output2 = torch::flatten(input, start_dim, end_dim);
            output2.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for invalid dimension combinations
        }
        
        // Test 3: Test with default parameters (start_dim=1, end_dim=-1)
        try {
            torch::nn::Flatten flatten_default;
            torch::Tensor output3 = flatten_default->forward(input);
            output3.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for 0-dim or 1-dim tensors
        }
        
        // Test 4: Flatten all dimensions (start_dim=0, end_dim=-1)
        try {
            torch::nn::Flatten flatten_all(
                torch::nn::FlattenOptions().start_dim(0).end_dim(-1)
            );
            torch::Tensor output4 = flatten_all->forward(input);
            output4.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for edge cases
        }
        
        // Test 5: Test with different data types
        try {
            torch::Tensor float_input = input.to(torch::kFloat32);
            torch::nn::Flatten flatten_float(
                torch::nn::FlattenOptions().start_dim(start_dim).end_dim(end_dim)
            );
            torch::Tensor output5 = flatten_float->forward(float_input);
            output5.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for invalid configurations
        }
        
        // Test 6: Test Unflatten as a related operation
        try {
            // Flatten first, then unflatten back
            torch::nn::Flatten flatten_for_unflatten(
                torch::nn::FlattenOptions().start_dim(0).end_dim(-1)
            );
            torch::Tensor flattened = flatten_for_unflatten->forward(input);
            
            // Unflatten back to original shape
            torch::nn::Unflatten unflatten_module(
                torch::nn::UnflattenOptions(0, input.sizes().vec())
            );
            torch::Tensor unflattened = unflatten_module->forward(flattened);
            unflattened.sum().item<float>();
        } catch (const c10::Error&) {
            // Expected for edge cases
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}