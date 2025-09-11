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
        
        // Create input tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.argwhere operation
        torch::Tensor result = torch::argwhere(input_tensor);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto sizes = result.sizes();
            auto dtype = result.dtype();
            
            // Force evaluation of the result
            auto accessor = result.accessor<int64_t, 2>();
            
            // Try to access the first element if available
            if (result.numel() > 0 && sizes[0] > 0 && sizes[1] > 0) {
                volatile int64_t first_element = accessor[0][0];
            }
        }
        
        // Try with different options if we have more data
        if (Size - offset >= 1 && offset < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Create a boolean mask from the original tensor
            torch::Tensor bool_mask = input_tensor.to(torch::kBool);
            
            // Apply argwhere on the boolean mask
            torch::Tensor bool_result = torch::argwhere(bool_mask);
            
            // Try with a tensor containing NaN values if the tensor is floating point
            if (at::isFloatingType(input_tensor.scalar_type())) {
                torch::Tensor nan_tensor = input_tensor.clone();
                
                // Insert some NaN values if tensor is not empty
                if (nan_tensor.numel() > 0) {
                    // Set first element to NaN
                    nan_tensor.flatten()[0] = std::numeric_limits<float>::quiet_NaN();
                    
                    // Apply argwhere on tensor with NaN
                    torch::Tensor nan_result = torch::argwhere(nan_tensor);
                }
            }
            
            // Try with a tensor containing infinity values if the tensor is floating point
            if (at::isFloatingType(input_tensor.scalar_type())) {
                torch::Tensor inf_tensor = input_tensor.clone();
                
                // Insert some infinity values if tensor is not empty
                if (inf_tensor.numel() > 0) {
                    // Set first element to infinity
                    inf_tensor.flatten()[0] = std::numeric_limits<float>::infinity();
                    
                    // Apply argwhere on tensor with infinity
                    torch::Tensor inf_result = torch::argwhere(inf_tensor);
                }
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
