#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create LogSigmoid module
        torch::nn::LogSigmoid logsigmoid;
        
        // Apply LogSigmoid to the input tensor
        torch::Tensor output = logsigmoid->forward(input);
        
        // Verify the output is valid
        if (output.numel() != input.numel()) {
            throw std::runtime_error("Output tensor has different number of elements than input tensor");
        }
        
        // Alternative way to apply LogSigmoid using functional API
        torch::Tensor output2 = torch::nn::functional::logsigmoid(input);
        
        // Verify both methods produce the same result
        if (!torch::allclose(output, output2)) {
            throw std::runtime_error("Module and functional implementations produced different results");
        }
        
        // Test with different input types
        if (offset + 1 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Apply LogSigmoid to the second input tensor
            torch::Tensor output3 = logsigmoid->forward(input2);
            
            // Verify the output is valid
            if (output3.numel() != input2.numel()) {
                throw std::runtime_error("Second output tensor has different number of elements than input tensor");
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