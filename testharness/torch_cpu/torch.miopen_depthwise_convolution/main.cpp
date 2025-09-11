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
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create weight tensor
        torch::Tensor weight;
        if (offset < Size) {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we've consumed all data, create a compatible weight tensor
            auto options = torch::TensorOptions().dtype(input.dtype());
            weight = torch::ones({1, input.size(1), 3, 3}, options);
        }
        
        // Create bias tensor (optional)
        torch::Tensor bias;
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        if (use_bias && offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Parse convolution parameters
        std::vector<int64_t> padding = {1, 1};
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> dilation = {1, 1};
        int64_t groups = 1;
        bool benchmark = false;
        bool deterministic = false;
        
        if (offset + 8 < Size) {
            padding[0] = static_cast<int64_t>(Data[offset++]) % 5;
            padding[1] = static_cast<int64_t>(Data[offset++]) % 5;
            stride[0] = static_cast<int64_t>(Data[offset++] % 3) + 1;
            stride[1] = static_cast<int64_t>(Data[offset++] % 3) + 1;
            dilation[0] = static_cast<int64_t>(Data[offset++] % 3) + 1;
            dilation[1] = static_cast<int64_t>(Data[offset++] % 3) + 1;
            
            if (offset < Size) {
                groups = static_cast<int64_t>(Data[offset++]) % 8 + 1;
            }
            if (offset < Size) {
                benchmark = Data[offset++] % 2 == 0;
            }
            if (offset < Size) {
                deterministic = Data[offset++] % 2 == 0;
            }
        }
        
        // Ensure input and weight have compatible shapes for convolution
        // Reshape if necessary to make the operation possible
        if (input.dim() < 4) {
            // Reshape to 4D tensor (N, C, H, W)
            std::vector<int64_t> new_shape = {1, 1, 1, 1};
            int64_t total_elements = input.numel();
            
            if (total_elements > 0) {
                if (input.dim() == 3) {
                    new_shape[0] = input.size(0);
                    new_shape[1] = input.size(1);
                    new_shape[2] = input.size(2);
                } else if (input.dim() == 2) {
                    new_shape[0] = 1;
                    new_shape[1] = input.size(0);
                    new_shape[2] = input.size(1);
                } else if (input.dim() == 1) {
                    new_shape[0] = 1;
                    new_shape[1] = 1;
                    new_shape[2] = input.size(0);
                }
            }
            
            input = input.reshape(new_shape);
        }
        
        if (weight.dim() < 4) {
            // Reshape to 4D tensor (out_channels, in_channels/groups, kH, kW)
            std::vector<int64_t> new_shape = {1, input.size(1), 1, 1};
            int64_t total_elements = weight.numel();
            
            if (total_elements > 0) {
                if (weight.dim() == 3) {
                    new_shape[0] = weight.size(0);
                    new_shape[2] = weight.size(1);
                    new_shape[3] = weight.size(2);
                } else if (weight.dim() == 2) {
                    new_shape[0] = weight.size(0);
                    new_shape[2] = weight.size(1);
                } else if (weight.dim() == 1) {
                    new_shape[0] = weight.size(0);
                }
            }
            
            weight = weight.reshape(new_shape);
        }
        
        // Ensure bias has compatible shape if used
        if (use_bias && bias.defined()) {
            if (bias.dim() > 1) {
                bias = bias.reshape({bias.numel()});
            }
            
            // Ensure bias has the right size (should match output channels)
            if (bias.size(0) != weight.size(0)) {
                bias = bias.repeat(weight.size(0) / (bias.size(0) > 0 ? bias.size(0) : 1));
                bias = bias.slice(0, 0, weight.size(0));
            }
        }
        
        // Adjust groups to be compatible with input and weight
        if (input.size(1) % groups != 0 || weight.size(0) % groups != 0) {
            // Find a valid groups value that divides both dimensions
            for (int64_t g = groups; g > 0; g--) {
                if (input.size(1) % g == 0 && weight.size(0) % g == 0) {
                    groups = g;
                    break;
                }
            }
        }
        
        // Ensure weight's in_channels dimension is compatible with input's channels and groups
        if (weight.size(1) != input.size(1) / groups) {
            auto options = torch::TensorOptions().dtype(input.dtype());
            weight = torch::ones({weight.size(0), input.size(1) / groups, weight.size(2), weight.size(3)}, options);
        }
        
        // Apply miopen_depthwise_convolution
        torch::Tensor output;
        
        if (use_bias && bias.defined()) {
            output = torch::miopen_depthwise_convolution(
                input, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
        } else {
            output = torch::miopen_depthwise_convolution(
                input, weight, {}, padding, stride, dilation, groups, benchmark, deterministic);
        }
        
        // Use the output to prevent optimization from removing the computation
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
