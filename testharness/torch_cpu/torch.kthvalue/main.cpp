#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with result

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract k value from the remaining data
        int64_t k_raw = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&k_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract dim value from the remaining data
        int64_t dim_raw = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim boolean from the remaining data
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Handle scalar tensors by reshaping
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
        // Skip empty tensors
        if (input.numel() == 0) {
            return 0;
        }
        
        // Ensure dim is within valid range for the tensor (supports negative indexing)
        int64_t dim = dim_raw % input.dim();
        
        // Ensure k is within valid range for the dimension (k is 1-indexed)
        int64_t dim_size = input.size(dim);
        if (dim_size <= 0) {
            return 0;
        }
        int64_t k = (std::abs(k_raw) % dim_size) + 1;
        
        // Test 1: Basic kthvalue operation
        {
            auto result = torch::kthvalue(input, k, dim, keepdim);
            
            // Access the values and indices to ensure they're computed
            auto values = std::get<0>(result);
            auto indices = std::get<1>(result);
            
            // Perform some operation on the results to ensure they're used
            (void)values.sum().item<float>();
            (void)indices.max().item<int64_t>();
        }
        
        // Test 2: kthvalue without explicit dim (uses last dimension)
        try {
            auto result = torch::kthvalue(input, k);
            auto values = std::get<0>(result);
            (void)values.sum().item<float>();
        } catch (...) {
            // May fail if k is too large for last dimension, that's expected
        }
        
        // Test 3: kthvalue with out arguments
        {
            // Pre-allocate output tensors with correct shapes
            auto expected_shape = input.sizes().vec();
            if (keepdim) {
                expected_shape[dim] = 1;
            } else {
                expected_shape.erase(expected_shape.begin() + dim);
            }
            
            torch::Tensor values_out = torch::empty(expected_shape, input.options());
            torch::Tensor indices_out = torch::empty(expected_shape, torch::dtype(torch::kLong).device(input.device()));
            
            torch::kthvalue_out(values_out, indices_out, input, k, dim, keepdim);
            
            // Verify outputs are populated
            (void)values_out.sum().item<float>();
            (void)indices_out.max().item<int64_t>();
        }
        
        // Test 4: Test with different k values if dimension is large enough
        if (dim_size > 1) {
            // Test with k=1 (minimum)
            try {
                auto result = torch::kthvalue(input, 1, dim, keepdim);
                (void)std::get<0>(result).sum().item<float>();
            } catch (...) {
                // Silently handle expected failures
            }
            
            // Test with k=dim_size (maximum)
            try {
                auto result = torch::kthvalue(input, dim_size, dim, keepdim);
                (void)std::get<0>(result).sum().item<float>();
            } catch (...) {
                // Silently handle expected failures
            }
        }
        
        // Test 5: Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2) {
            try {
                auto transposed = input.transpose(0, 1);
                int64_t t_dim = dim_raw % transposed.dim();
                int64_t t_dim_size = transposed.size(t_dim);
                if (t_dim_size > 0) {
                    int64_t t_k = (std::abs(k_raw) % t_dim_size) + 1;
                    auto result = torch::kthvalue(transposed, t_k, t_dim, keepdim);
                    (void)std::get<0>(result).sum().item<float>();
                }
            } catch (...) {
                // Silently handle expected failures
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