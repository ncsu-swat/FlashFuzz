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
        
        // Apply torch.corrcoef operation
        torch::Tensor result = torch::corrcoef(input_tensor);
        
        // Try with different input types
        if (input_tensor.dtype() != torch::kFloat32 && input_tensor.dtype() != torch::kFloat64) {
            // Convert to float for numerical stability
            torch::Tensor float_tensor = input_tensor.to(torch::kFloat32);
            torch::Tensor result_float = torch::corrcoef(float_tensor);
        }
        
        // Try with empty tensor
        if (offset + 1 < Size) {
            std::vector<int64_t> empty_shape;
            if (Data[offset] % 3 == 0) {
                empty_shape = {0};
            } else if (Data[offset] % 3 == 1) {
                empty_shape = {0, 2};
            } else {
                empty_shape = {2, 0};
            }
            
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            try {
                torch::Tensor result_empty = torch::corrcoef(empty_tensor);
            } catch (const std::exception&) {
                // Expected exception for empty tensor, continue
            }
        }
        
        // Try with 1D tensor
        if (input_tensor.dim() == 1 && input_tensor.size(0) > 0) {
            torch::Tensor result_1d = torch::corrcoef(input_tensor);
        }
        
        // Try with tensor containing NaN/Inf values
        if (offset + 1 < Size) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            torch::Tensor special_tensor;
            
            if (Data[offset] % 3 == 0) {
                // Create tensor with NaN
                special_tensor = torch::ones({2, 3}, options);
                special_tensor.index_put_({0, 0}, std::numeric_limits<float>::quiet_NaN());
            } else if (Data[offset] % 3 == 1) {
                // Create tensor with Inf
                special_tensor = torch::ones({2, 3}, options);
                special_tensor.index_put_({0, 0}, std::numeric_limits<float>::infinity());
            } else {
                // Create tensor with -Inf
                special_tensor = torch::ones({2, 3}, options);
                special_tensor.index_put_({0, 0}, -std::numeric_limits<float>::infinity());
            }
            
            try {
                torch::Tensor result_special = torch::corrcoef(special_tensor);
            } catch (const std::exception&) {
                // May throw for invalid values, continue
            }
        }
        
        // Try with high-dimensional tensor
        if (input_tensor.dim() > 2) {
            try {
                torch::Tensor result_high_dim = torch::corrcoef(input_tensor);
            } catch (const std::exception&) {
                // May throw for invalid dimensions, continue
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
