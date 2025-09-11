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
            weight = torch::ones({1, input.size(0), 1, 1}, options);
        }
        
        // Parse convolution parameters from remaining data
        int64_t stride_h = 1, stride_w = 1;
        int64_t padding_h = 0, padding_w = 0;
        int64_t dilation_h = 1, dilation_w = 1;
        int64_t groups = 1;
        
        if (offset + 7 <= Size) {
            stride_h = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            stride_w = static_cast<int64_t>(Data[offset++]) % 4 + 1;
            padding_h = static_cast<int64_t>(Data[offset++]) % 3;
            padding_w = static_cast<int64_t>(Data[offset++]) % 3;
            dilation_h = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            dilation_w = static_cast<int64_t>(Data[offset++]) % 3 + 1;
            groups = static_cast<int64_t>(Data[offset++]) % 4 + 1;
        }
        
        // Ensure tensors are on CUDA if available
        if (torch::cuda::is_available()) {
            input = input.cuda();
            weight = weight.cuda();
        }
        
        // Ensure input and weight have proper dimensions for convolution
        if (input.dim() < 3) {
            // Reshape to at least 3D tensor (N, C, D)
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                new_shape = {1, 1, 1};
            } else if (input.dim() == 1) {
                new_shape = {1, 1, input.size(0)};
            } else { // dim == 2
                new_shape = {1, input.size(0), input.size(1)};
            }
            input = input.reshape(new_shape);
        }
        
        if (weight.dim() < 4) {
            // Reshape to 4D tensor (out_channels, in_channels/groups, kH, kW)
            std::vector<int64_t> new_shape;
            if (weight.dim() == 0) {
                new_shape = {1, 1, 1, 1};
            } else if (weight.dim() == 1) {
                new_shape = {1, 1, 1, weight.size(0)};
            } else if (weight.dim() == 2) {
                new_shape = {1, 1, weight.size(0), weight.size(1)};
            } else { // dim == 3
                new_shape = {1, weight.size(0), weight.size(1), weight.size(2)};
            }
            weight = weight.reshape(new_shape);
        }
        
        // Ensure weight's in_channels is compatible with input's channels and groups
        int64_t in_channels = input.size(1);
        if (weight.size(1) * groups != in_channels) {
            // Adjust groups to be compatible
            if (in_channels > 0) {
                groups = std::gcd(in_channels, weight.size(1));
                if (groups == 0) groups = 1;
            } else {
                groups = 1;
            }
        }
        
        // Create stride, padding, and dilation vectors
        std::vector<int64_t> stride = {stride_h, stride_w};
        std::vector<int64_t> padding = {padding_h, padding_w};
        std::vector<int64_t> dilation = {dilation_h, dilation_w};
        
        // Apply cudnn_convolution_relu with bias parameter (set to nullopt)
        torch::Tensor output = torch::cudnn_convolution_relu(
            input, weight, std::nullopt,
            stride, padding, dilation, groups
        );
        
        // Use the output to prevent optimization from removing the operation
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
