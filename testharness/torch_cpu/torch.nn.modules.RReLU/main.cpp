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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for RReLU from the remaining data
        float lower = 0.125;
        float upper = 0.3333;
        bool inplace = false;
        
        // If we have more data, use it to set parameters
        if (offset + 8 <= Size) {
            // Extract lower bound (between 0 and 1)
            float raw_lower;
            std::memcpy(&raw_lower, Data + offset, sizeof(float));
            offset += sizeof(float);
            lower = std::abs(raw_lower) / (std::abs(raw_lower) + 1.0f); // Normalize to [0,1]
            
            // Extract upper bound (between lower and 1)
            float raw_upper;
            std::memcpy(&raw_upper, Data + offset, sizeof(float));
            offset += sizeof(float);
            upper = lower + (1.0f - lower) * (std::abs(raw_upper) / (std::abs(raw_upper) + 1.0f));
            
            // Extract inplace flag
            if (offset < Size) {
                inplace = Data[offset++] & 0x01;
            }
        }
        
        // Create RReLU module
        torch::nn::RReLU rrelu = torch::nn::RReLU(
            torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(inplace)
        );
        
        // Set to training mode to test the randomized behavior
        rrelu->train();
        torch::Tensor output_train = rrelu->forward(input);
        
        // Test in eval mode too
        rrelu->eval();
        torch::Tensor output_eval = rrelu->forward(input);
        
        // Test the functional version as well
        torch::Tensor output_functional = torch::rrelu(input, lower, upper, inplace);
        
        // Test with different generator
        auto generator = torch::Generator();
        torch::Tensor output_with_generator = torch::rrelu(input, lower, upper, false, generator);
        
        // Test backward pass if possible
        if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
            input.set_requires_grad(true);
            auto output = rrelu->forward(input);
            
            // Sum to get scalar for backward
            if (output.numel() > 0) {
                auto sum = output.sum();
                sum.backward();
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
