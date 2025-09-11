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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse alpha parameter if we have more data
        double alpha = 1.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse inplace parameter if we have more data
        bool inplace = false;
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create ELU module with the parsed parameters
        torch::nn::ELU elu_module(torch::nn::ELUOptions().alpha(alpha).inplace(inplace));
        
        // Apply the ELU operation
        torch::Tensor output = elu_module->forward(input);
        
        // Verify the output is valid
        if (output.numel() != input.numel()) {
            throw std::runtime_error("Output tensor has different number of elements than input");
        }
        
        // Try a different alpha value if we have more data
        if (offset + sizeof(double) <= Size) {
            double new_alpha;
            std::memcpy(&new_alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Create a new ELU module with the new alpha
            torch::nn::ELU elu_module2(torch::nn::ELUOptions().alpha(new_alpha).inplace(inplace));
            torch::Tensor output2 = elu_module2->forward(input);
        }
        
        // Try with inplace=true if we originally had inplace=false
        if (!inplace) {
            torch::nn::ELU elu_module_inplace(torch::nn::ELUOptions().alpha(alpha).inplace(true));
            
            // Clone the input since we're using inplace operation
            torch::Tensor input_clone = input.clone();
            torch::Tensor output_inplace = elu_module_inplace->forward(input_clone);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
