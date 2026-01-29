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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // cummax requires at least 1 dimension
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Get a dimension to perform cummax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dim is within valid range [-dim(), dim()-1]
            int64_t ndim = input_tensor.dim();
            dim = (dim % ndim + ndim) % ndim;  // Normalize to [0, ndim-1]
            
            // Randomly make it negative to test negative indexing
            if (offset < Size && (Data[offset] & 1)) {
                dim = dim - ndim;  // Convert to negative dimension
            }
        }
        
        // Apply cummax operation
        try {
            std::tuple<torch::Tensor, torch::Tensor> result = torch::cummax(input_tensor, dim);
            
            // Access the values and indices from the result
            torch::Tensor values = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Perform some operations on the results to ensure they're used
            if (values.numel() > 0 && indices.numel() > 0) {
                volatile float dummy = values.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error&) {
            // Expected errors for invalid dimensions or inputs
        }
        
        // Test with different dimensions
        for (int64_t test_dim = 0; test_dim < input_tensor.dim(); test_dim++) {
            try {
                auto [vals, idxs] = torch::cummax(input_tensor, test_dim);
                
                // Verify output shapes match input
                if (vals.numel() > 0) {
                    volatile float v = vals.sum().item<float>();
                    (void)v;
                }
            } catch (const c10::Error&) {
                // Expected for some inputs
            }
        }
        
        // Test with negative dimension
        try {
            auto [vals_neg, idxs_neg] = torch::cummax(input_tensor, -1);
            if (vals_neg.numel() > 0) {
                volatile float v = vals_neg.sum().item<float>();
                (void)v;
            }
        } catch (const c10::Error&) {
            // Expected for some inputs
        }
        
        // Test edge case: if tensor is empty but has dimensions
        if (input_tensor.numel() == 0 && input_tensor.dim() > 0) {
            for (int64_t test_dim = 0; test_dim < input_tensor.dim(); test_dim++) {
                try {
                    auto [empty_values, empty_indices] = torch::cummax(input_tensor, test_dim);
                } catch (const c10::Error&) {
                    // Expected for empty tensors
                }
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