#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
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
            return 0;
        }
        
        // Create bias tensor (optional)
        torch::Tensor bias;
        bool use_bias = offset < Size && Data[offset++] % 2 == 0;
        if (use_bias && offset < Size) {
            bias = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Parse stride
        std::vector<int64_t> stride;
        if (offset < Size) {
            uint8_t stride_size = Data[offset++] % 3 + 1; // 1-3 dimensions
            for (uint8_t i = 0; i < stride_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t s;
                std::memcpy(&s, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                stride.push_back(std::abs(s) % 4 + 1); // 1-4 stride
            }
        } else {
            stride = {1};
        }
        
        // Parse padding
        std::vector<int64_t> padding;
        if (offset < Size) {
            uint8_t padding_size = Data[offset++] % 3 + 1; // 1-3 dimensions
            for (uint8_t i = 0; i < padding_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t p;
                std::memcpy(&p, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                padding.push_back(std::abs(p) % 3); // 0-2 padding
            }
        } else {
            padding = {0};
        }
        
        // Parse dilation
        std::vector<int64_t> dilation;
        if (offset < Size) {
            uint8_t dilation_size = Data[offset++] % 3 + 1; // 1-3 dimensions
            for (uint8_t i = 0; i < dilation_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t d;
                std::memcpy(&d, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dilation.push_back(std::abs(d) % 3 + 1); // 1-3 dilation
            }
        } else {
            dilation = {1};
        }
        
        // Parse transposed flag
        bool transposed = offset < Size && Data[offset++] % 2 == 0;
        
        // Parse groups
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            groups = std::abs(groups) % 4 + 1; // 1-4 groups
        }
        
        // Apply convolution operation
        torch::Tensor output;
        try {
            output = torch::convolution(
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                transposed,
                {0}, // output_padding (using default)
                groups
            );
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and handled
            return 0;
        }
        
        // Perform some operations on the output to ensure it's used
        if (output.defined()) {
            auto sum = output.sum();
            if (sum.item<float>() == -1.0f) {
                // This is just to prevent the compiler from optimizing away the sum calculation
                return 1;
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
