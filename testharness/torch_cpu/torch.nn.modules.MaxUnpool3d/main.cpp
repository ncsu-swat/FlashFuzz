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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create indices tensor
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers
        indices = indices.to(torch::kInt64);
        
        // Extract parameters for MaxUnpool3d
        int64_t kernel_size_d = 2;
        int64_t kernel_size_h = 2;
        int64_t kernel_size_w = 2;
        int64_t stride_d = 2;
        int64_t stride_h = 2;
        int64_t stride_w = 2;
        int64_t padding_d = 0;
        int64_t padding_h = 0;
        int64_t padding_w = 0;
        
        // Extract parameters from input data if available
        if (offset + 9 < Size) {
            kernel_size_d = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            kernel_size_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            kernel_size_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            stride_d = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            stride_h = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            stride_w = static_cast<int64_t>(Data[offset++]) % 5 + 1;
            padding_d = static_cast<int64_t>(Data[offset++]) % 3;
            padding_h = static_cast<int64_t>(Data[offset++]) % 3;
            padding_w = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Create output_size tensor if there's enough data
        std::vector<int64_t> output_size;
        if (input.dim() >= 5 && offset + 3 < Size) {
            output_size.push_back(input.size(0));  // Batch size
            output_size.push_back(input.size(1));  // Channels
            output_size.push_back(static_cast<int64_t>(Data[offset++]) % 32 + 1);  // Depth
            output_size.push_back(static_cast<int64_t>(Data[offset++]) % 32 + 1);  // Height
            output_size.push_back(static_cast<int64_t>(Data[offset++]) % 32 + 1);  // Width
        }
        
        // Create MaxUnpool3d module
        torch::nn::MaxUnpool3d unpool = torch::nn::MaxUnpool3d(
            torch::nn::MaxUnpool3dOptions(
                {kernel_size_d, kernel_size_h, kernel_size_w})
                .stride({stride_d, stride_h, stride_w})
                .padding({padding_d, padding_h, padding_w})
        );
        
        // Apply MaxUnpool3d
        torch::Tensor output;
        if (!output_size.empty()) {
            output = unpool->forward(input, indices, output_size);
        } else {
            output = unpool->forward(input, indices);
        }
        
        // Use the output to prevent optimization
        if (output.defined()) {
            volatile auto sum = output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}