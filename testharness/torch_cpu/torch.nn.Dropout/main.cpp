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
        
        // Need at least 3 bytes for basic operation
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dropout probability from the input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Extract train flag from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Extract inplace flag from the input data
        bool inplace = false; // Default value
        if (offset < Size) {
            inplace = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Create Dropout module
        torch::nn::Dropout dropout_module(torch::nn::DropoutOptions().p(p).inplace(inplace));
        
        // Set the module to training or evaluation mode
        if (train) {
            dropout_module->train();
        } else {
            dropout_module->eval();
        }
        
        // Apply dropout to the input tensor
        torch::Tensor output = dropout_module->forward(input);
        
        // Test the functional interface as well
        torch::Tensor output_functional = torch::dropout(input, p, train);
        
        // Test with edge case probabilities if we have more data
        if (offset + sizeof(float) <= Size) {
            float edge_p;
            std::memcpy(&edge_p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Try with extreme probability values
            torch::nn::Dropout dropout_edge(torch::nn::DropoutOptions().p(edge_p).inplace(inplace));
            if (train) {
                dropout_edge->train();
            } else {
                dropout_edge->eval();
            }
            torch::Tensor output_edge = dropout_edge->forward(input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
