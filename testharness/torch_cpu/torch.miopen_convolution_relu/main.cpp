#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
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
            // If we don't have enough data, create a simple weight tensor
            weight = torch::ones({1, input.size(1), 3, 3});
        }
        
        // Create bias tensor (optional)
        torch::Tensor bias;
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        if (use_bias && offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        } else if (use_bias) {
            // Create a simple bias if we need one but don't have data
            bias = torch::ones({weight.size(0)});
        }
        
        // Parse convolution parameters
        std::vector<int64_t> stride = {1, 1};
        std::vector<int64_t> padding = {0, 0};
        std::vector<int64_t> dilation = {1, 1};
        int64_t groups = 1;
        
        // If we have more data, use it to set parameters
        if (offset + 5 < Size) {
            stride[0] = (Data[offset] % 3) + 1;
            stride[1] = (Data[offset+1] % 3) + 1;
            padding[0] = Data[offset+2] % 3;
            padding[1] = Data[offset+3] % 3;
            dilation[0] = (Data[offset+4] % 2) + 1;
            dilation[1] = (Data[offset+5] % 2) + 1;
            offset += 6;
        }
        
        if (offset < Size) {
            groups = (Data[offset++] % 4) + 1;
        }
        
        // Ensure input and weight have compatible dimensions for convolution
        if (input.dim() >= 3 && weight.dim() >= 4) {
            // Try to make dimensions compatible
            if (input.size(0) == 0) {
                input = torch::ones({1, input.size(1), input.size(2), input.size(3)});
            }
            
            // Ensure weight has compatible dimensions with input
            if (weight.size(1) * groups != input.size(1)) {
                weight = torch::ones({weight.size(0), input.size(1) / groups, weight.size(2), weight.size(3)});
            }
            
            // Ensure bias has compatible dimensions with weight
            if (use_bias && bias.dim() > 0 && bias.size(0) != weight.size(0)) {
                bias = torch::ones({weight.size(0)});
            }
            
            // Apply miopen_convolution_relu
            torch::Tensor output;
            if (use_bias) {
                output = torch::miopen_convolution_relu(
                    input, weight, bias, stride, padding, dilation, groups);
            } else {
                output = torch::miopen_convolution_relu(
                    input, weight, {}, stride, padding, dilation, groups);
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