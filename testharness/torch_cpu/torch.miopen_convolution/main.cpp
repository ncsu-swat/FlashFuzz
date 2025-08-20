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
            // If we don't have enough data, create a compatible weight tensor
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
        std::vector<int64_t> stride, padding, dilation;
        int64_t groups = 1;
        
        // Parse stride
        if (offset + 1 < Size) {
            int64_t stride_size = (Data[offset++] % 3) + 1; // 1-3 values
            for (int i = 0; i < stride_size && offset < Size; i++) {
                stride.push_back((Data[offset++] % 5) + 1); // 1-5 stride
            }
        }
        if (stride.empty()) stride = {1, 1};
        
        // Parse padding
        if (offset + 1 < Size) {
            int64_t padding_size = (Data[offset++] % 3) + 1; // 1-3 values
            for (int i = 0; i < padding_size && offset < Size; i++) {
                padding.push_back(Data[offset++] % 3); // 0-2 padding
            }
        }
        if (padding.empty()) padding = {0, 0};
        
        // Parse dilation
        if (offset + 1 < Size) {
            int64_t dilation_size = (Data[offset++] % 3) + 1; // 1-3 values
            for (int i = 0; i < dilation_size && offset < Size; i++) {
                dilation.push_back((Data[offset++] % 3) + 1); // 1-3 dilation
            }
        }
        if (dilation.empty()) dilation = {1, 1};
        
        // Parse groups
        if (offset < Size) {
            groups = (Data[offset++] % 4) + 1; // 1-4 groups
        }
        
        // Benchmark flag (always false for fuzzing)
        bool benchmark = false;
        
        // Deterministic flag
        bool deterministic = offset < Size && (Data[offset++] % 2 == 0);
        
        // Ensure input has at least 3 dimensions (N, C, spatial dims...)
        if (input.dim() < 3) {
            auto shape = input.sizes().vec();
            while (shape.size() < 3) {
                shape.push_back(1);
            }
            input = input.reshape(shape);
        }
        
        // Ensure weight has at least 4 dimensions (out_channels, in_channels/groups, spatial dims...)
        if (weight.dim() < 4) {
            auto shape = weight.sizes().vec();
            while (shape.size() < 4) {
                shape.push_back(1);
            }
            weight = weight.reshape(shape);
        }
        
        // Ensure bias has correct shape if used
        if (use_bias && bias.defined()) {
            if (bias.dim() != 1 || bias.size(0) != weight.size(0)) {
                bias = torch::ones({weight.size(0)}, bias.options());
            }
        }
        
        // Apply miopen_convolution
        torch::Tensor output;
        if (use_bias && bias.defined()) {
            output = torch::miopen_convolution(
                input, weight, bias,
                padding, stride, dilation,
                groups, benchmark, deterministic
            );
        } else {
            output = torch::miopen_convolution(
                input, weight, {},
                padding, stride, dilation,
                groups, benchmark, deterministic
            );
        }
        
        // Force evaluation of the output tensor
        output.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        return 0; // discard the input
    }
    return 0; // keep the input
}