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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special.erfcx operation
        torch::Tensor result = torch::special::erfcx(input);
        
        // Try some edge cases with modified tensors if we have enough data
        if (offset + 1 < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_input;
            
            uint8_t selector = Data[offset++];
            if (selector % 4 == 0) {
                // Very large positive values
                extreme_input = input * 1e10;
            } else if (selector % 4 == 1) {
                // Very large negative values
                extreme_input = input * -1e10;
            } else if (selector % 4 == 2) {
                // Values close to zero
                extreme_input = input * 1e-10;
            } else {
                // NaN and Inf values
                extreme_input = input.clone();
                if (extreme_input.numel() > 0) {
                    // Set some elements to NaN or Inf if possible
                    if (extreme_input.is_floating_point()) {
                        auto flat = extreme_input.flatten();
                        if (flat.numel() > 0) flat[0] = std::numeric_limits<float>::infinity();
                        if (flat.numel() > 1) flat[1] = -std::numeric_limits<float>::infinity();
                        if (flat.numel() > 2) flat[2] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
            
            // Apply erfcx to the extreme input
            torch::Tensor extreme_result = torch::special::erfcx(extreme_input);
        }
        
        // Try with different tensor options if we have more data
        if (offset + 1 < Size) {
            uint8_t option_selector = Data[offset++];
            
            // Create a view or strided tensor
            if (input.numel() > 0 && input.dim() > 0) {
                torch::Tensor modified_input;
                
                if (option_selector % 3 == 0 && input.dim() > 1) {
                    // Transpose the tensor
                    modified_input = input.transpose(0, input.dim() - 1);
                } else if (option_selector % 3 == 1) {
                    // Create a non-contiguous slice
                    modified_input = input;
                    for (int64_t d = 0; d < input.dim(); d++) {
                        if (input.size(d) > 1) {
                            modified_input = modified_input.slice(d, 0, -1, 2);
                            break;
                        }
                    }
                } else {
                    // Create a view with different shape
                    if (input.numel() > 0) {
                        modified_input = input.reshape({-1});
                    } else {
                        modified_input = input;
                    }
                }
                
                // Apply erfcx to the modified input
                torch::Tensor modified_result = torch::special::erfcx(modified_input);
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