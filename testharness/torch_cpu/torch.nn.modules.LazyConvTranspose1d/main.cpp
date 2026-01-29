#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 3 dimensions for ConvTranspose1d (N, C, L)
        if (input.dim() < 3) {
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0;
            }
            input = input.reshape({1, 1, numel});
        }
        
        // Ensure input is float type for convolution
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get in_channels from input
        int64_t in_channels = input.size(1);
        
        // Parse out_channels
        int64_t out_channels = 1;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            out_channels = std::abs(val) % 16 + 1;
        }
        
        // Parse kernel_size
        int64_t kernel_size = 1;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            kernel_size = std::abs(val) % 7 + 1;
        }
        
        // Parse stride
        int64_t stride = 1;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            stride = std::abs(val) % 5 + 1;
        }
        
        // Parse padding
        int64_t padding = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            padding = std::abs(val) % 4;
        }
        
        // Parse dilation
        int64_t dilation = 1;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            dilation = std::abs(val) % 3 + 1;
        }
        
        // Parse output_padding - must be less than max(stride, dilation)
        int64_t output_padding = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            int64_t max_output_padding = std::max(stride, dilation) - 1;
            if (max_output_padding > 0) {
                output_padding = std::abs(val) % (max_output_padding + 1);
            }
        }
        
        // Parse groups - must divide both in_channels and out_channels
        int64_t groups = 1;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t val;
            std::memcpy(&val, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            // Find valid divisors of both in_channels and out_channels
            int64_t candidate = std::abs(val) % 4 + 1;
            while (candidate > 1 && (in_channels % candidate != 0 || out_channels % candidate != 0)) {
                candidate--;
            }
            groups = candidate;
        }
        
        // Parse bias flag
        bool bias = true;
        if (offset < Size) {
            bias = Data[offset++] & 0x1;
        }
        
        // Create ConvTranspose1d module (C++ frontend doesn't have Lazy variant)
        // ConvTranspose1d requires explicit in_channels
        auto options = torch::nn::ConvTranspose1dOptions(in_channels, out_channels, kernel_size)
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .groups(groups)
            .bias(bias)
            .dilation(dilation);
        
        auto module = torch::nn::ConvTranspose1d(options);
        
        // Forward pass
        try {
            torch::Tensor output = module->forward(input);
            
            // Access some properties of the output to ensure computation
            auto output_size = output.sizes();
            volatile float output_sum = output.sum().item<float>();
            (void)output_sum;
            (void)output_size;
        } catch (const c10::Error&) {
            // Shape mismatches or other PyTorch errors are expected
            return 0;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}