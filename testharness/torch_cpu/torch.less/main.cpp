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
            // If no more data, create a tensor with same shape but different values
            tensor2 = tensor1.clone();
            
            // Try to modify tensor2 to make it different from tensor1
            if (tensor2.numel() > 0) {
                // Add a small value to make tensors different
                if (tensor2.is_floating_point()) {
                    tensor2.add_(0.5);
                } else if (tensor2.dtype() == torch::kBool) {
                    tensor2 = ~tensor2;
                } else {
                    tensor2.add_(1);
                }
            }
        }
        
        // Apply torch.less operation (torch::lt is the C++ equivalent)
        torch::Tensor result = torch::lt(tensor1, tensor2);
        
        // Also test the method form
        torch::Tensor result_method = tensor1.lt(tensor2);
        
        // Test torch::less alias if available
        torch::Tensor result_less = torch::less(tensor1, tensor2);
        
        // Try scalar versions if tensor has elements
        if (tensor1.numel() > 0) {
            try {
                // Use a simple scalar instead of extracting from tensor
                // This avoids issues with multi-element tensors
                torch::Tensor result_scalar_int = tensor1.lt(1);
                torch::Tensor result_scalar_float = tensor1.lt(0.5);
                
                // Test with negative values
                torch::Tensor result_scalar_neg = tensor1.lt(-1);
            } catch (const std::exception&) {
                // Scalar comparison might fail for some dtypes
            }
        }
        
        // Test with out parameter
        try {
            torch::Tensor out_tensor = torch::empty_like(tensor1, torch::kBool);
            torch::lt_out(out_tensor, tensor1, tensor2);
        } catch (const std::exception&) {
            // May fail due to shape/dtype mismatches
        }
        
        // Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0}, tensor1.options());
            torch::Tensor result_empty = torch::lt(empty_tensor, empty_tensor);
        } catch (const std::exception&) {
            // This might throw, which is fine
        }
        
        // Try with tensors of different dtypes (type promotion)
        try {
            torch::Tensor int_tensor = tensor1.to(torch::kInt32);
            torch::Tensor float_tensor = tensor2.to(torch::kFloat32);
            torch::Tensor result_mixed = torch::lt(int_tensor, float_tensor);
        } catch (const std::exception&) {
            // This might throw, which is fine
        }
        
        // Try with boolean tensors
        try {
            torch::Tensor bool_tensor1 = tensor1.to(torch::kBool);
            torch::Tensor bool_tensor2 = tensor2.to(torch::kBool);
            torch::Tensor result_bool = torch::lt(bool_tensor1, bool_tensor2);
        } catch (const std::exception&) {
            // Boolean comparison might have restrictions
        }
        
        // Test broadcasting explicitly with known compatible shapes
        try {
            torch::Tensor t1 = torch::randn({2, 3});
            torch::Tensor t2 = torch::randn({3});
            torch::Tensor result_broadcast = torch::lt(t1, t2);
        } catch (const std::exception&) {
            // Broadcasting test
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // Keep the input
}