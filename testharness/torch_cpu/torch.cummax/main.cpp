#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to perform cummax along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                // Allow negative dimensions for testing edge cases
                dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            }
        }
        
        // Apply cummax operation
        std::tuple<torch::Tensor, torch::Tensor> result = torch::cummax(input_tensor, dim);
        
        // Access the values and indices from the result
        torch::Tensor values = std::get<0>(result);
        torch::Tensor indices = std::get<1>(result);
        
        // Perform some operations on the results to ensure they're used
        if (!values.sizes().empty() && !indices.sizes().empty()) {
            auto sum_values = values.sum();
            auto max_indices = indices.max();
            
            // Prevent compiler from optimizing away the operations
            if (sum_values.numel() > 0 && max_indices.numel() > 0) {
                volatile float dummy = sum_values.item<float>() + max_indices.item<float>();
                (void)dummy;
            }
        }
        
        // Try another variant with named return values
        auto [values2, indices2] = torch::cummax(input_tensor, dim);
        
        // Test edge case: if tensor is empty but has dimensions
        if (input_tensor.numel() == 0 && input_tensor.dim() > 0) {
            for (int64_t test_dim = 0; test_dim < input_tensor.dim(); test_dim++) {
                auto [empty_values, empty_indices] = torch::cummax(input_tensor, test_dim);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
