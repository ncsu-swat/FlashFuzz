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
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            return 0;
        }
        
        // Create weight tensor
        torch::Tensor weight;
        try {
            weight = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            return 0;
        }
        
        // Create z tensor (z tensor in cudnn_convolution_add_relu)
        torch::Tensor z;
        try {
            z = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception &e) {
            return 0;
        }
        
        // Parse convolution parameters
        std::vector<int64_t> padding;
        std::vector<int64_t> stride;
        std::vector<int64_t> dilation;
        int64_t groups = 1;
        
        // Parse padding
        if (offset + 2 <= Size) {
            uint8_t padding_size = Data[offset++] % 3 + 1; // 1-3 values
            for (uint8_t i = 0; i < padding_size && offset < Size; i++) {
                padding.push_back(static_cast<int64_t>(Data[offset++]) % 4); // 0-3 padding
            }
        }
        
        // Parse stride
        if (offset + 2 <= Size) {
            uint8_t stride_size = Data[offset++] % 3 + 1; // 1-3 values
            for (uint8_t i = 0; i < stride_size && offset < Size; i++) {
                stride.push_back(static_cast<int64_t>(Data[offset++]) % 3 + 1); // 1-3 stride
            }
        }
        
        // Parse dilation
        if (offset + 2 <= Size) {
            uint8_t dilation_size = Data[offset++] % 3 + 1; // 1-3 values
            for (uint8_t i = 0; i < dilation_size && offset < Size; i++) {
                dilation.push_back(static_cast<int64_t>(Data[offset++]) % 3 + 1); // 1-3 dilation
            }
        }
        
        // Parse groups
        if (offset < Size) {
            groups = static_cast<int64_t>(Data[offset++]) % 4 + 1; // 1-4 groups
        }
        
        // Ensure tensors are on CUDA if available
        if (torch::cuda::is_available()) {
            input = input.cuda();
            weight = weight.cuda();
            z = z.cuda();
        }
        
        // Ensure input and weight have compatible data types for cudnn
        auto supported_dtypes = {torch::kFloat, torch::kHalf, torch::kDouble};
        bool is_supported_dtype = false;
        for (auto dtype : supported_dtypes) {
            if (input.dtype() == dtype) {
                is_supported_dtype = true;
                break;
            }
        }
        
        if (!is_supported_dtype) {
            input = input.to(torch::kFloat);
            weight = weight.to(torch::kFloat);
            z = z.to(torch::kFloat);
        }
        
        // Apply cudnn_convolution_add_relu
        try {
            torch::Tensor output = torch::cudnn_convolution_add_relu(
                input, weight, z, std::nullopt, std::nullopt, stride, padding, dilation, groups);
        } catch (const std::exception &e) {
            // Expected exceptions from invalid inputs are fine
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
