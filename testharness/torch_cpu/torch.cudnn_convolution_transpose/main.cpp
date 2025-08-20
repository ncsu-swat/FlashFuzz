#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Check if we have enough data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a compatible weight tensor
            auto options = torch::TensorOptions().dtype(input.dtype());
            weight = torch::ones({1, input.size(0), 3, 3}, options);
        }
        
        // Parse convolution parameters from remaining data
        std::vector<int64_t> padding = {1, 1};
        std::vector<int64_t> output_padding = {0, 0};
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> dilation = {1, 1};
        int64_t groups = 1;
        bool benchmark = false;
        bool deterministic = false;
        bool allow_tf32 = false;
        
        // If we have more data, use it to set parameters
        if (offset + 8 < Size) {
            // Parse padding
            int64_t pad_h = static_cast<int64_t>(Data[offset++]) % 4;
            int64_t pad_w = static_cast<int64_t>(Data[offset++]) % 4;
            padding = {pad_h, pad_w};
            
            // Parse output_padding
            int64_t out_pad_h = static_cast<int64_t>(Data[offset++]) % 3;
            int64_t out_pad_w = static_cast<int64_t>(Data[offset++]) % 3;
            output_padding = {out_pad_h, out_pad_w};
            
            // Parse stride
            int64_t stride_h = (static_cast<int64_t>(Data[offset++]) % 3) + 1;
            int64_t stride_w = (static_cast<int64_t>(Data[offset++]) % 3) + 1;
            stride = {stride_h, stride_w};
            
            // Parse dilation
            int64_t dilation_h = (static_cast<int64_t>(Data[offset++]) % 2) + 1;
            int64_t dilation_w = (static_cast<int64_t>(Data[offset++]) % 2) + 1;
            dilation = {dilation_h, dilation_w};
            
            // Parse groups
            if (offset < Size) {
                groups = (static_cast<int64_t>(Data[offset++]) % 4) + 1;
            }
            
            // Parse boolean flags
            if (offset < Size) {
                uint8_t flags = Data[offset++];
                benchmark = (flags & 0x01) != 0;
                deterministic = (flags & 0x02) != 0;
                allow_tf32 = (flags & 0x04) != 0;
            }
        }
        
        // Try to move tensors to CUDA if available
        if (torch::cuda::is_available()) {
            input = input.cuda();
            weight = weight.cuda();
        }
        
        // Apply cudnn_convolution_transpose
        torch::Tensor output;
        
        // Ensure input and weight have compatible dimensions for convolution
        if (input.dim() >= 3 && weight.dim() >= 3) {
            // Reshape tensors if needed to make them compatible
            if (input.dim() == 3) {
                input = input.unsqueeze(0);  // Add batch dimension
            }
            
            if (weight.dim() == 3) {
                weight = weight.unsqueeze(0);  // Add output channel dimension
            }
            
            // Ensure weight has proper shape for transposed convolution
            // For transposed conv: weight shape should be [in_channels, out_channels/groups, ...]
            if (weight.size(0) % groups != 0) {
                // Adjust groups to be compatible with weight shape
                groups = 1;
            }
            
            // Call the cudnn_convolution_transpose function
            output = torch::cudnn_convolution_transpose(
                input, weight, padding, output_padding, stride, dilation, 
                groups, benchmark, deterministic, allow_tf32
            );
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        return 0; // keep the input
    }
}