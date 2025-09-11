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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.t_copy operation
        torch::Tensor result = torch::t_copy(input_tensor);
        
        // Verify the operation completed by accessing some property
        auto sizes = result.sizes();
        
        // Optionally, verify the transpose was done correctly
        // For 2D tensors, dimensions should be swapped
        if (input_tensor.dim() == 2) {
            auto input_sizes = input_tensor.sizes();
            if (sizes[0] != input_sizes[1] || sizes[1] != input_sizes[0]) {
                throw std::runtime_error("Transpose dimensions incorrect");
            }
        }
        // For 1D tensors, result should be 2D with first dim = 1
        else if (input_tensor.dim() == 1) {
            auto input_sizes = input_tensor.sizes();
            if (sizes[0] != input_sizes[0] || sizes[1] != 1) {
                throw std::runtime_error("1D tensor transpose incorrect");
            }
        }
        // For 0D tensors, result should still be 0D
        else if (input_tensor.dim() == 0) {
            if (result.dim() != 0) {
                throw std::runtime_error("0D tensor transpose incorrect");
            }
        }
        // For higher dimensions, t_copy should throw an exception
        // which will be caught by our outer try-catch
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
