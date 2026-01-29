#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout

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
        
        // Need at least a few bytes to work with
        if (Size < 4) {
            return 0;
        }
        
        // Determine number of tensors to stack (2-8)
        uint8_t num_tensors = (Data[offset++] % 7) + 2;
        
        // Create first tensor to establish the shape
        torch::Tensor first_tensor;
        try {
            first_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // Can't create base tensor, discard input
            return 0;
        }
        
        // Get the shape of the first tensor
        auto shape = first_tensor.sizes().vec();
        auto dtype = first_tensor.dtype();
        
        // Create a vector to hold our tensors with matching shapes
        std::vector<torch::Tensor> tensors;
        tensors.push_back(first_tensor);
        
        // Create additional tensors with the same shape
        for (uint8_t i = 1; i < num_tensors; ++i) {
            try {
                // Create tensor with same shape as first tensor
                torch::Tensor tensor = torch::rand(shape, dtype);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If we can't create a tensor, continue with what we have
                break;
            }
        }
        
        // Need at least two tensors to stack meaningfully
        if (tensors.size() < 2) {
            return 0;
        }
        
        // Get dimension to stack along from fuzzer data
        int64_t dim = 0;
        if (offset < Size) {
            int8_t dim_value;
            std::memcpy(&dim_value, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            // Constrain dim to valid range: [-ndim-1, ndim]
            int64_t ndim = static_cast<int64_t>(tensors[0].dim());
            if (ndim > 0) {
                dim = dim_value % (ndim + 1);
            } else {
                dim = 0;
            }
        }
        
        // Apply torch.stack operation
        try {
            torch::Tensor result = torch::stack(tensors, dim);
            // Verify the result has expected properties
            (void)result.sizes();
            (void)result.numel();
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and not a bug
        }
        
        // Test with TensorList directly
        try {
            torch::TensorList tensor_list(tensors);
            torch::Tensor result = torch::stack(tensor_list, dim);
            (void)result.dim();
        } catch (const c10::Error& e) {
            // Expected for invalid configurations
        }
        
        // Try stacking along different dimensions
        if (offset < Size) {
            try {
                int8_t alt_dim_raw;
                std::memcpy(&alt_dim_raw, Data + offset, sizeof(int8_t));
                offset += sizeof(int8_t);
                
                // Test with dimension 0 (always valid)
                torch::Tensor result0 = torch::stack(tensors, 0);
                (void)result0.size(0);
                
                // Test with last dimension
                int64_t last_dim = static_cast<int64_t>(tensors[0].dim());
                torch::Tensor result_last = torch::stack(tensors, last_dim);
                (void)result_last.sizes();
                
                // Test with negative dimension
                torch::Tensor result_neg = torch::stack(tensors, -1);
                (void)result_neg.numel();
                
            } catch (const c10::Error& e) {
                // Expected for edge cases
            }
        }
        
        // Test with single tensor (edge case)
        try {
            std::vector<torch::Tensor> single_tensor = {tensors[0]};
            torch::Tensor result = torch::stack(single_tensor, 0);
            (void)result.sizes();
        } catch (const c10::Error& e) {
            // May or may not be allowed
        }
        
        // Test with empty tensor vector (should throw)
        try {
            std::vector<torch::Tensor> empty_tensors;
            torch::Tensor result = torch::stack(empty_tensors, 0);
            (void)result;
        } catch (const c10::Error& e) {
            // Expected to throw for empty input
        }
        
        // Test stacking tensors of different dtypes (will be converted or error)
        if (tensors.size() >= 2) {
            try {
                std::vector<torch::Tensor> mixed_tensors;
                mixed_tensors.push_back(tensors[0].to(torch::kFloat32));
                mixed_tensors.push_back(tensors[1].to(torch::kFloat64));
                torch::Tensor result = torch::stack(mixed_tensors, 0);
                (void)result.dtype();
            } catch (const c10::Error& e) {
                // May throw due to dtype mismatch
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