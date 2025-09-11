#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a parameter tensor for weight normalization
        torch::Tensor param_tensor;
        if (offset < Size - 2) {
            param_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            param_tensor = torch::randn_like(input_tensor);
        }
        
        // Get a byte to determine which nn.utils function to test
        uint8_t function_selector = 0;
        if (offset < Size) {
            function_selector = Data[offset++];
        }
        
        // Test various torch.nn.utils functions
        switch (function_selector % 4) {
            case 0: {
                // Test clip_grad_norm_
                std::vector<torch::Tensor> parameters = {input_tensor, param_tensor};
                double max_norm = 1.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&max_norm, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                double norm_type = 2.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&norm_type, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                torch::nn::utils::clip_grad_norm_(parameters, max_norm, norm_type);
                break;
            }
            
            case 1: {
                // Test clip_grad_value_
                std::vector<torch::Tensor> parameters = {input_tensor, param_tensor};
                double clip_value = 1.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&clip_value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                torch::nn::utils::clip_grad_value_(parameters, clip_value);
                break;
            }
            
            case 2: {
                // Test parameters_to_vector
                std::vector<torch::Tensor> parameters = {input_tensor, param_tensor};
                torch::Tensor vec = torch::nn::utils::parameters_to_vector(parameters);
                break;
            }
            
            case 3: {
                // Test vector_to_parameters
                torch::Tensor vec;
                if (input_tensor.dim() == 1) {
                    vec = input_tensor;
                } else {
                    vec = torch::flatten(input_tensor);
                }
                std::vector<torch::Tensor> parameters = {param_tensor};
                torch::nn::utils::vector_to_parameters(vec, parameters);
                break;
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
