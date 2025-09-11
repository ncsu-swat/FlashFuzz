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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a model parameter tensor
        torch::Tensor param = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a vector of parameters
        std::vector<torch::Tensor> parameters;
        parameters.push_back(param);
        
        // Try to create another parameter if we have enough data
        if (offset + 4 < Size) {
            torch::Tensor param2 = fuzzer_utils::createTensor(Data, Size, offset);
            parameters.push_back(param2);
        }
        
        // Get a byte to determine whether to use vector_to_parameters or parameters_to_vector
        bool use_vector_to_parameters = true;
        if (offset < Size) {
            use_vector_to_parameters = Data[offset++] & 0x1;
        }
        
        if (use_vector_to_parameters) {
            // First convert parameters to vector
            torch::Tensor flat_param = torch::nn::utils::parameters_to_vector(parameters);
            
            // Then convert vector back to parameters
            torch::nn::utils::vector_to_parameters(flat_param, parameters);
        } else {
            // Convert parameters to vector
            torch::Tensor flat_param = torch::nn::utils::parameters_to_vector(parameters);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
