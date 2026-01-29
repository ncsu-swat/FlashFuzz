#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Create second tensor if we have more data
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, use the same tensor or a scalar
            if (Size % 2 == 0) {
                tensor2 = tensor1.clone();
            } else {
                // Create a scalar tensor
                tensor2 = torch::tensor(1.0, tensor1.options());
            }
        }
        
        // Try different variants of the gt operation
        try {
            // Element-wise comparison: tensor > tensor
            torch::Tensor result1 = torch::gt(tensor1, tensor2);
            (void)result1;
        } catch (const std::exception& e) {
            // Silently catch expected failures (shape mismatches, etc.)
        }
        
        try {
            // Tensor > scalar
            double scalar_value = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            // Sanitize NaN/Inf to avoid undefined behavior
            if (std::isnan(scalar_value) || std::isinf(scalar_value)) {
                scalar_value = 0.0;
            }
            torch::Tensor result2 = torch::gt(tensor1, scalar_value);
            (void)result2;
        } catch (const std::exception& e) {
            // Silently catch expected failures
        }
        
        try {
            // Scalar > tensor (create scalar tensor first)
            double scalar_value = 1.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            }
            if (std::isnan(scalar_value) || std::isinf(scalar_value)) {
                scalar_value = 1.0;
            }
            torch::Tensor scalar_tensor = torch::tensor(scalar_value, tensor1.options());
            torch::Tensor result3 = torch::gt(scalar_tensor, tensor1);
            (void)result3;
        } catch (const std::exception& e) {
            // Silently catch expected failures
        }
        
        try {
            // Using the operator overload
            torch::Tensor result4 = tensor1 > tensor2;
            (void)result4;
        } catch (const std::exception& e) {
            // Silently catch expected failures
        }
        
        try {
            // Using out variant - first do the comparison to get the result shape,
            // then use gt_out with a pre-allocated tensor of the same shape
            torch::Tensor temp_result = torch::gt(tensor1, tensor2);
            torch::Tensor out = torch::empty_like(temp_result);
            torch::gt_out(out, tensor1, tensor2);
            (void)out;
        } catch (const std::exception& e) {
            // Silently catch expected failures
        }
        
        // Try with different tensor types if we have enough data
        if (offset + 2 < Size) {
            try {
                // Create tensors with different dtypes - int vs float comparison
                torch::Tensor int_tensor = tensor1.to(torch::kInt);
                torch::Tensor float_tensor = tensor2.to(torch::kFloat);
                torch::Tensor result6 = torch::gt(int_tensor, float_tensor);
                (void)result6;
            } catch (const std::exception& e) {
                // Silently catch expected failures
            }
            
            try {
                // Long tensor comparison
                torch::Tensor long_tensor = tensor1.to(torch::kLong);
                torch::Tensor result7 = torch::gt(long_tensor, tensor2);
                (void)result7;
            } catch (const std::exception& e) {
                // Silently catch expected failures
            }
        }
        
        // Test with contiguous vs non-contiguous tensors
        try {
            if (tensor1.dim() >= 2 && tensor1.size(0) > 1 && tensor1.size(1) > 1) {
                torch::Tensor transposed = tensor1.transpose(0, 1);
                torch::Tensor result8 = torch::gt(transposed, tensor2);
                (void)result8;
            }
        } catch (const std::exception& e) {
            // Silently catch expected failures
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}