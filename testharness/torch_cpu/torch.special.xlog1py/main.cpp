#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for torch.special.xlog1py
        // This function computes x * log1p(y) with special handling for x=0, y=-1
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Not enough data for second tensor, use a simple tensor instead
            torch::Tensor y = torch::ones_like(x);
            
            // Apply the operation
            torch::Tensor result = torch::special::xlog1py(x, y);
            
            // Force evaluation of the tensor
            result.sum().item<float>();
            
            return 0;
        }
        
        // Create the second tensor
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the operation - broadcasting is handled automatically by PyTorch
        torch::Tensor result = torch::special::xlog1py(x, y);
        
        // Force evaluation of the tensor
        result.sum().item<float>();
        
        // Test with output tensor (out parameter version)
        try {
            torch::Tensor out = torch::empty_like(result);
            torch::special::xlog1py_out(out, x, y);
            out.sum().item<float>();
        } catch (...) {
            // Shape mismatch or other expected errors - ignore silently
        }
        
        // Try with swapped arguments to explore different code paths
        try {
            torch::Tensor swapped_result = torch::special::xlog1py(y, x);
            swapped_result.sum().item<float>();
        } catch (...) {
            // Shape mismatch possible - ignore silently
        }
        
        // Test tensor-scalar operation if we have extra data
        if (offset < Size) {
            // Use data byte to create a scalar value
            float scalar_value = static_cast<float>(Data[offset]) / 255.0f * 10.0f - 5.0f;
            
            // Try with scalar as second argument
            try {
                torch::Tensor scalar_result = torch::special::xlog1py(x, torch::scalar_tensor(scalar_value));
                scalar_result.sum().item<float>();
            } catch (...) {
                // Ignore errors
            }
            
            // Try with scalar as first argument (as scalar tensor)
            try {
                torch::Tensor scalar_x = torch::scalar_tensor(scalar_value);
                torch::Tensor scalar_result2 = torch::special::xlog1py(scalar_x, y);
                scalar_result2.sum().item<float>();
            } catch (...) {
                // Ignore errors
            }
        }
        
        // Test with different dtypes
        try {
            torch::Tensor x_double = x.to(torch::kDouble);
            torch::Tensor y_double = y.to(torch::kDouble);
            torch::Tensor double_result = torch::special::xlog1py(x_double, y_double);
            double_result.sum().item<double>();
        } catch (...) {
            // Dtype conversion issues - ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;
}