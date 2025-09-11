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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a dimension value from the remaining data if available
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create GLU module
        torch::nn::GLUOptions options;
        
        // If we have enough data, set the dimension
        if (dim != -1 && input.dim() > 0) {
            // Normalize dim to be within valid range [-input.dim(), input.dim()-1]
            dim = dim % (2 * input.dim());
            if (dim < 0) {
                dim += input.dim();
            }
            options.dim(dim);
        }
        
        auto glu = torch::nn::GLU(options);
        
        // Apply GLU to the input tensor
        torch::Tensor output = glu->forward(input);
        
        // Ensure the output is valid by accessing some property
        auto output_sizes = output.sizes();
        
        // Try to access elements to ensure no segfaults
        if (output.numel() > 0) {
            auto first_element = output.flatten()[0];
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
