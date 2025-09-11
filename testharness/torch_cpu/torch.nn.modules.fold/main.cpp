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
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract fold parameters from the remaining data
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Extract output_size
        int64_t output_height = 0;
        int64_t output_width = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_height, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_height = std::abs(output_height) % 100 + 1; // Ensure positive
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_width, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            output_width = std::abs(output_width) % 100 + 1; // Ensure positive
        }
        
        // Extract kernel_size
        int64_t kernel_height = 0;
        int64_t kernel_width = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_height, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_height = std::abs(kernel_height) % 10 + 1; // Ensure positive
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&kernel_width, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            kernel_width = std::abs(kernel_width) % 10 + 1; // Ensure positive
        }
        
        // Extract stride
        int64_t stride_height = 0;
        int64_t stride_width = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_height, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_height = std::abs(stride_height) % 5 + 1; // Ensure positive
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&stride_width, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            stride_width = std::abs(stride_width) % 5 + 1; // Ensure positive
        }
        
        // Extract padding
        int64_t padding_height = 0;
        int64_t padding_width = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_height, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_height = std::abs(padding_height) % 5; // Can be zero
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_width, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding_width = std::abs(padding_width) % 5; // Can be zero
        }
        
        // Extract dilation
        int64_t dilation_height = 0;
        int64_t dilation_width = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation_height, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation_height = std::abs(dilation_height) % 3 + 1; // Ensure positive
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dilation_width, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            dilation_width = std::abs(dilation_width) % 3 + 1; // Ensure positive
        }
        
        // Create fold options
        torch::nn::FoldOptions options(
            {kernel_height, kernel_width},
            {output_height, output_width}
        );
        
        options.stride({stride_height, stride_width})
               .padding({padding_height, padding_width})
               .dilation({dilation_height, dilation_width});
        
        // Create fold module
        torch::nn::Fold fold_module(options);
        
        // Apply fold operation
        torch::Tensor output = fold_module->forward(input);
        
        // Try to access output properties to ensure computation is complete
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
