#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for torch.special.erfc
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.erfc operation
        torch::Tensor result = torch::special::erfc(input);
        
        // Try some edge cases with modified tensors if we have enough data
        if (offset + 1 < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_input;
            
            uint8_t selector = Data[offset++];
            if (selector % 4 == 0) {
                // Very large values
                extreme_input = input * 1e10;
            } else if (selector % 4 == 1) {
                // Very small values
                extreme_input = input * 1e-10;
            } else if (selector % 4 == 2) {
                // NaN values where supported
                if (input.is_floating_point() || input.is_complex()) {
                    extreme_input = input / 0.0;
                } else {
                    extreme_input = input;
                }
            } else {
                // Inf values where supported
                if (input.is_floating_point() || input.is_complex()) {
                    extreme_input = input.abs() / 0.0;
                } else {
                    extreme_input = input;
                }
            }
            
            // Apply torch.special.erfc to the extreme input
            torch::Tensor extreme_result = torch::special::erfc(extreme_input);
        }
        
        // Try with different tensor views if possible
        if (!input.sizes().empty() && input.numel() > 1) {
            // Create a view with different strides
            std::vector<int64_t> new_shape;
            int64_t total_elements = 1;
            
            for (int i = 0; i < input.dim(); i++) {
                if (input.size(i) > 1) {
                    new_shape.push_back(input.size(i));
                    total_elements *= input.size(i);
                }
            }
            
            if (!new_shape.empty() && total_elements == input.numel()) {
                torch::Tensor reshaped = input.reshape(new_shape);
                torch::Tensor result_reshaped = torch::special::erfc(reshaped);
            }
            
            // Try with transposed tensor if it has at least 2 dimensions
            if (input.dim() >= 2) {
                torch::Tensor transposed = input.transpose(0, input.dim() - 1);
                torch::Tensor result_transposed = torch::special::erfc(transposed);
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