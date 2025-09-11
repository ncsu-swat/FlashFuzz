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
            // If we don't have enough data, create a compatible weight tensor
            auto options = torch::TensorOptions().dtype(input.dtype());
            weight = torch::ones({1, input.size(0), 3, 3}, options);
        }
        
        // Create bias tensor (optional)
        torch::Tensor bias;
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        if (use_bias && offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Parse convolution parameters
        std::vector<int64_t> stride, padding, output_padding, dilation;
        int64_t groups = 1;
        
        // Parse stride
        if (offset + 1 < Size) {
            uint8_t stride_size = Data[offset++] % 3 + 1;
            for (int i = 0; i < stride_size && offset < Size; i++) {
                stride.push_back((Data[offset++] % 3) + 1);
            }
        } else {
            stride = {1, 1};
        }
        
        // Parse padding
        if (offset + 1 < Size) {
            uint8_t padding_size = Data[offset++] % 3 + 1;
            for (int i = 0; i < padding_size && offset < Size; i++) {
                padding.push_back(Data[offset++] % 3);
            }
        } else {
            padding = {0, 0};
        }
        
        // Parse output_padding
        if (offset + 1 < Size) {
            uint8_t output_padding_size = Data[offset++] % 3 + 1;
            for (int i = 0; i < output_padding_size && offset < Size; i++) {
                output_padding.push_back(Data[offset++] % 3);
            }
        } else {
            output_padding = {0, 0};
        }
        
        // Parse dilation
        if (offset + 1 < Size) {
            uint8_t dilation_size = Data[offset++] % 3 + 1;
            for (int i = 0; i < dilation_size && offset < Size; i++) {
                dilation.push_back((Data[offset++] % 2) + 1);
            }
        } else {
            dilation = {1, 1};
        }
        
        // Parse groups
        if (offset < Size) {
            groups = (Data[offset++] % 4) + 1;
        }
        
        // Parse benchmark flag
        bool benchmark = offset < Size && Data[offset++] % 2 == 0;
        
        // Parse deterministic flag
        bool deterministic = offset < Size && Data[offset++] % 2 == 0;
        
        // Ensure tensors are on the same device (CPU for fuzzing)
        input = input.to(torch::kCPU);
        weight = weight.to(torch::kCPU);
        if (use_bias) {
            bias = bias.to(torch::kCPU);
        }
        
        // Ensure input and weight have compatible dimensions for convolution
        if (input.dim() < 3 || weight.dim() < 3) {
            // Reshape tensors to have at least 3 dimensions
            if (input.dim() < 3) {
                std::vector<int64_t> new_shape(3, 1);
                for (int i = 0; i < input.dim(); i++) {
                    new_shape[i] = input.size(i);
                }
                input = input.reshape(new_shape);
            }
            
            if (weight.dim() < 3) {
                std::vector<int64_t> new_shape(4, 1);
                for (int i = 0; i < weight.dim(); i++) {
                    new_shape[i] = weight.size(i);
                }
                weight = weight.reshape(new_shape);
            }
        }
        
        // Ensure weight has proper shape for convolution
        if (weight.dim() != 4) {
            weight = weight.reshape({1, 1, 3, 3});
        }
        
        // Ensure bias has proper shape if used
        if (use_bias) {
            if (bias.dim() != 1 || bias.size(0) != weight.size(1)) {
                bias = torch::ones({weight.size(1)}, bias.options());
            }
        }
        
        // Apply miopen_convolution_transpose
        torch::Tensor output;
        try {
            output = torch::miopen_convolution_transpose(
                input, weight, bias, padding, output_padding, stride, dilation, groups,
                benchmark, deterministic
            );
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<float>() == -1.0f) {
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}
