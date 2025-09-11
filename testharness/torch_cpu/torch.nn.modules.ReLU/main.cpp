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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create ReLU module with different configurations
        bool inplace = false;
        if (offset < Size) {
            inplace = Data[offset++] & 0x1; // Use 1 bit to determine inplace
        }
        
        // Create ReLU module
        torch::nn::ReLU relu_module(torch::nn::ReLUOptions().inplace(inplace));
        
        // Apply ReLU operation
        torch::Tensor output = relu_module->forward(input);
        
        // Test functional version as well
        torch::Tensor output_functional = torch::relu(input);
        
        // Test with threshold parameter (ReLU6)
        double threshold = 6.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&threshold, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply threshold ReLU (ReLU6-like)
        torch::Tensor output_threshold = torch::clamp(input, 0.0, threshold);
        
        // Test with leaky ReLU parameters
        double negative_slope = 0.01;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&negative_slope, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply leaky ReLU
        torch::Tensor output_leaky = torch::leaky_relu(input, negative_slope);
        
        // Test edge cases with different tensor types
        if (offset < Size) {
            // Create another tensor with different dtype if possible
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor output2 = relu_module->forward(input2);
        }
        
        // Test with zero-sized dimensions
        std::vector<int64_t> empty_shape = {0, 2, 3};
        torch::Tensor empty_tensor = torch::empty(empty_shape);
        torch::Tensor empty_output = relu_module->forward(empty_tensor);
        
        // Test with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(-5.0);
        torch::Tensor scalar_output = relu_module->forward(scalar_tensor);
        
        // Test with boolean tensor
        torch::Tensor bool_tensor = torch::tensor(true);
        torch::Tensor bool_output = relu_module->forward(bool_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
