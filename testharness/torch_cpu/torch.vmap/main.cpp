#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and function parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for vmap from the remaining data
        uint8_t in_dims = 0;
        uint8_t out_dims = 0;
        
        if (offset + 2 <= Size) {
            in_dims = Data[offset++];
            out_dims = Data[offset++];
        }
        
        // Ensure in_dims and out_dims are within reasonable range
        in_dims = in_dims % 4;  // Limit to 0-3
        out_dims = out_dims % 4; // Limit to 0-3
        
        // Create a simple function to be vmapped
        auto func = [](const torch::Tensor& x) -> torch::Tensor {
            return x.sin();
        };
        
        // Apply vmap with different configurations
        torch::Tensor result;
        
        // Try different vmap configurations based on the input data
        if (offset < Size) {
            uint8_t vmap_config = Data[offset++] % 4;
            
            switch (vmap_config) {
                case 0:
                    // Basic vmap usage
                    result = torch::func::vmap(func, in_dims)(input_tensor);
                    break;
                    
                case 1:
                    // vmap with specified out_dims
                    result = torch::func::vmap(func, in_dims, out_dims)(input_tensor);
                    break;
                    
                case 2: {
                    // vmap with a function that takes multiple tensors
                    auto multi_tensor_func = [](const torch::Tensor& x, const torch::Tensor& y) -> torch::Tensor {
                        return x + y;
                    };
                    
                    // Create a second tensor if we have enough data
                    if (offset < Size) {
                        torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        result = torch::func::vmap(multi_tensor_func, in_dims)(input_tensor, second_tensor);
                    } else {
                        // Not enough data for second tensor, use the same tensor twice
                        result = torch::func::vmap(multi_tensor_func, in_dims)(input_tensor, input_tensor);
                    }
                    break;
                }
                
                case 3: {
                    // vmap with a nested vmap
                    auto nested_func = [in_dims](const torch::Tensor& x) -> torch::Tensor {
                        auto inner_func = [](const torch::Tensor& y) -> torch::Tensor {
                            return y.cos();
                        };
                        return torch::func::vmap(inner_func, 0)(x);
                    };
                    
                    result = torch::func::vmap(nested_func, in_dims)(input_tensor);
                    break;
                }
            }
        } else {
            // Default case if we don't have enough data for configuration
            result = torch::func::vmap(func, in_dims)(input_tensor);
        }
        
        // Force evaluation of the result to ensure any errors are triggered
        result.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}