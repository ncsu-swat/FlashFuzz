#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <limits>         // For numeric_limits

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
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor with remaining data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, create one with same shape
            tensor2 = torch::ones_like(tensor1);
        }
        
        // Test element-wise comparison (with or without broadcasting)
        try {
            torch::Tensor result = torch::lt(tensor1, tensor2);
        } catch (const std::exception&) {
            // May fail due to incompatible shapes for broadcasting
        }
        
        // Also test the method form
        try {
            torch::Tensor result = tensor1.lt(tensor2);
        } catch (const std::exception&) {
            // May fail due to incompatible shapes for broadcasting
        }
        
        // Test scalar comparison using a fuzz-derived scalar value
        if (Size > offset) {
            float scalar_val = static_cast<float>(Data[offset % Size]) - 128.0f;
            torch::Scalar scalar(scalar_val);
            
            // Test tensor < scalar (function form)
            torch::Tensor result1 = torch::lt(tensor1, scalar);
            
            // Test tensor < scalar (method form)
            torch::Tensor result2 = tensor1.lt(scalar);
        }
        
        // Test with output tensor
        try {
            torch::Tensor output = torch::empty_like(tensor1, torch::kBool);
            torch::lt_out(output, tensor1, tensor2);
        } catch (const std::exception&) {
            // May fail due to shape mismatch
        }
        
        // Test with different dtypes - convert tensor2 to different type
        try {
            torch::Tensor tensor2_int = tensor2.to(torch::kInt32);
            torch::Tensor result = torch::lt(tensor1, tensor2_int);
        } catch (const std::exception&) {
            // May fail for certain dtype combinations
        }
        
        // Test with NaN values if floating point
        if (tensor1.is_floating_point()) {
            torch::Tensor nan_tensor = torch::full_like(tensor1, std::numeric_limits<float>::quiet_NaN());
            torch::Tensor result = torch::lt(tensor1, nan_tensor);
            
            // Also test NaN in first operand
            torch::Tensor result2 = torch::lt(nan_tensor, tensor2);
        }
        
        // Test with infinity values if floating point
        if (tensor1.is_floating_point()) {
            torch::Tensor pos_inf_tensor = torch::full_like(tensor1, std::numeric_limits<float>::infinity());
            torch::Tensor neg_inf_tensor = torch::full_like(tensor1, -std::numeric_limits<float>::infinity());
            
            torch::Tensor result1 = torch::lt(tensor1, pos_inf_tensor);
            torch::Tensor result2 = torch::lt(neg_inf_tensor, tensor1);
        }
        
        // Test comparing tensor with itself (should be all false)
        torch::Tensor self_compare = torch::lt(tensor1, tensor1);
        
        // Test with contiguous vs non-contiguous tensors
        if (tensor1.dim() >= 2 && tensor1.size(0) > 0 && tensor1.size(1) > 0) {
            try {
                torch::Tensor transposed = tensor1.transpose(0, 1);
                torch::Tensor result = torch::lt(transposed, transposed);
            } catch (const std::exception&) {
                // May fail for certain shapes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}