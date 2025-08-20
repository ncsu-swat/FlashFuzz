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
        
        // Create indices tensor (same shape as input)
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are valid (integers)
        indices = indices.to(torch::kInt64);
        
        // Parse kernel size
        std::vector<int64_t> kernel_size;
        for (int i = 0; i < 3 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t k;
            std::memcpy(&k, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure kernel size is positive
            kernel_size.push_back(std::abs(k) % 8 + 1);
        }
        
        // If we don't have enough data for kernel_size, use default values
        while (kernel_size.size() < 3) {
            kernel_size.push_back(2);
        }
        
        // Parse stride
        std::vector<int64_t> stride;
        for (int i = 0; i < 3 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t s;
            std::memcpy(&s, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure stride is positive
            stride.push_back(std::abs(s) % 4 + 1);
        }
        
        // If we don't have enough data for stride, use default values
        while (stride.size() < 3) {
            stride.push_back(kernel_size[stride.size()]);
        }
        
        // Parse padding
        std::vector<int64_t> padding;
        for (int i = 0; i < 3 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t p;
            std::memcpy(&p, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Allow padding to be zero or positive
            padding.push_back(std::abs(p) % 4);
        }
        
        // If we don't have enough data for padding, use default values
        while (padding.size() < 3) {
            padding.push_back(0);
        }
        
        // Parse output_size (optional)
        std::vector<int64_t> output_size;
        if (offset + 1 < Size && Data[offset++] % 2 == 0) {  // 50% chance to use output_size
            for (int i = 0; i < 5 && offset + sizeof(int64_t) <= Size; i++) {
                int64_t os;
                std::memcpy(&os, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                // Ensure output size is positive
                output_size.push_back(std::abs(os) % 32 + 1);
            }
        }
        
        // Create MaxUnpool3d module
        torch::nn::MaxUnpool3d unpool = torch::nn::MaxUnpool3d(
            torch::nn::MaxUnpool3dOptions(kernel_size)
                .stride(stride)
                .padding(padding)
        );
        
        // Apply MaxUnpool3d
        torch::Tensor output;
        if (output_size.empty()) {
            output = unpool->forward(input, indices);
        } else {
            output = unpool->forward(input, indices, output_size);
        }
        
        // Use the output to prevent it from being optimized away
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