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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply selu operation
        torch::Tensor output = torch::selu(input);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should try inplace version
            bool use_inplace = Data[offset++] & 0x1;
            
            if (use_inplace) {
                // Create a copy for inplace operation
                torch::Tensor input_copy = input.clone();
                input_copy = torch::selu(input_copy);
            }
        }
        
        // Try with different alpha and scale parameters if we have more data
        if (offset + 8 < Size) {
            // Extract 8 bytes for alpha and scale parameters
            double alpha_raw, scale_raw;
            std::memcpy(&alpha_raw, Data + offset, sizeof(double));
            offset += sizeof(double);
            std::memcpy(&scale_raw, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure alpha and scale are valid (non-zero, finite)
            double alpha = std::isfinite(alpha_raw) ? alpha_raw : 1.6732632423543772848170429916717;
            double scale = std::isfinite(scale_raw) ? scale_raw : 1.0507009873554804934193349852946;
            
            // Apply custom selu with extracted parameters
            // Note: PyTorch C++ API doesn't directly expose custom alpha/scale for selu,
            // so we implement the formula manually: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
            torch::Tensor pos_part = torch::relu(input);
            torch::Tensor neg_part = torch::min(torch::zeros_like(input), alpha * (torch::exp(input) - 1.0));
            torch::Tensor custom_selu = scale * (pos_part + neg_part);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
