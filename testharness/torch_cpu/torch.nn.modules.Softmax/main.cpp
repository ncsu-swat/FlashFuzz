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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension for softmax
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create Softmax module with different dimensions
        // No need to check if dim is valid - let PyTorch handle it
        torch::nn::Softmax softmax_module(dim);
        
        // Apply softmax operation
        torch::Tensor output = softmax_module->forward(input_tensor);
        
        // Try with different dimensions if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t dim2;
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::nn::Softmax softmax_module2(dim2);
            torch::Tensor output2 = softmax_module2->forward(input_tensor);
        }
        
        // Try with default dimension (last dimension)
        torch::nn::Softmax default_softmax(-1);
        torch::Tensor default_output = default_softmax->forward(input_tensor);
        
        // Try with extreme dimension values
        if (input_tensor.dim() > 0) {
            // Negative dimension (should wrap around)
            torch::nn::Softmax neg_softmax(-input_tensor.dim());
            torch::Tensor neg_output = neg_softmax->forward(input_tensor);
            
            // Very large dimension (should be handled by PyTorch)
            torch::nn::Softmax large_softmax(1000000);
            torch::Tensor large_output = large_softmax->forward(input_tensor);
        }
        
        // Try with different dtypes if we have a floating point tensor
        if (input_tensor.is_floating_point()) {
            // Try with half precision
            torch::Tensor half_input = input_tensor.to(torch::kHalf);
            torch::nn::Softmax half_softmax(dim);
            torch::Tensor half_output = half_softmax->forward(half_input);
            
            // Try with double precision
            torch::Tensor double_input = input_tensor.to(torch::kDouble);
            torch::nn::Softmax double_softmax(dim);
            torch::Tensor double_output = double_softmax->forward(double_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
