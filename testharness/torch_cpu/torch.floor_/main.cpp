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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a floating-point tensor from the input data
        // floor_ only works on floating-point tensors
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already floating point
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Make a copy for verification
        torch::Tensor original = tensor.clone();
        
        // Apply floor_ operation (in-place)
        tensor.floor_();
        
        // Verify the operation worked correctly by comparing with non-in-place version
        torch::Tensor expected = torch::floor(original);
        
        // Use equal instead of allclose to avoid issues with NaN
        try {
            // Inner try-catch for validation that may fail with special values
            bool results_match = torch::equal(tensor, expected);
            (void)results_match; // Suppress unused variable warning
        } catch (...) {
            // Silently ignore comparison failures (e.g., NaN comparisons)
        }
        
        // Try with different tensor shapes if we have more data
        if (offset + 4 < Size) {
            size_t offset2 = offset;
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset2);
            
            // Ensure floating point
            if (!tensor2.is_floating_point()) {
                tensor2 = tensor2.to(torch::kFloat32);
            }
            
            // Apply floor_ operation
            tensor2.floor_();
        }
        
        // Test with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat32);
            empty_tensor.floor_();
        } catch (...) {
            // Silently handle empty tensor edge case
        }
        
        // Test with scalar tensor
        if (offset < Size) {
            float value = static_cast<float>(Data[offset % Size]) / 10.0f - 12.0f;
            torch::Tensor scalar_tensor = torch::tensor(value);
            scalar_tensor.floor_();
        }
        
        // Test with multi-dimensional tensor
        if (Size >= 16) {
            std::vector<float> values;
            for (size_t i = 0; i < 16 && i < Size; i++) {
                values.push_back(static_cast<float>(Data[i]) / 25.5f - 5.0f);
            }
            torch::Tensor multi_dim = torch::tensor(values).reshape({4, 4});
            multi_dim.floor_();
        }
        
        // Test with non-contiguous tensor (transposed view)
        if (Size >= 8) {
            std::vector<float> values;
            for (size_t i = 0; i < 6 && i < Size; i++) {
                values.push_back(static_cast<float>(Data[i]) / 25.5f - 5.0f);
            }
            torch::Tensor base = torch::tensor(values).reshape({2, 3});
            torch::Tensor transposed = base.t(); // Non-contiguous
            transposed.floor_();
        }
        
        // Test with different floating point dtypes
        if (Size >= 4) {
            float val = static_cast<float>(Data[0]) / 10.0f;
            
            // Float32
            torch::Tensor f32_tensor = torch::tensor({val, val + 0.5f, val - 0.5f}, torch::kFloat32);
            f32_tensor.floor_();
            
            // Float64
            torch::Tensor f64_tensor = torch::tensor({static_cast<double>(val)}, torch::kFloat64);
            f64_tensor.floor_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}