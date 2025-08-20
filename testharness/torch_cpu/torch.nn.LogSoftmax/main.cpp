#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for dimension selection
        if (Size < 1) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension for LogSoftmax
        int64_t dim = 0;
        if (offset < Size) {
            // Extract a dimension value from the input data
            uint8_t dim_byte = Data[offset++];
            
            // If tensor has dimensions, select one of them
            if (input.dim() > 0) {
                dim = dim_byte % input.dim();
            }
        }
        
        // Create LogSoftmax module with the selected dimension
        torch::nn::LogSoftmax log_softmax((torch::nn::LogSoftmaxOptions(dim)));
        
        // Apply LogSoftmax to the input tensor
        torch::Tensor output = log_softmax->forward(input);
        
        // Try with functional interface as well
        torch::Tensor output2 = torch::log_softmax(input, dim);
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}