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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for RReLU from the remaining data
        double lower = 0.125;
        double upper = 0.3333333333333333;
        bool inplace = false;
        
        if (offset + 8 <= Size) {
            // Extract lower bound
            std::memcpy(&lower, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + 8 <= Size) {
            // Extract upper bound
            std::memcpy(&upper, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset < Size) {
            // Extract inplace flag
            inplace = Data[offset] & 0x1;
            offset++;
        }
        
        // Create RReLU module
        torch::nn::RReLU rrelu = torch::nn::RReLU(
            torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(inplace)
        );
        
        // Apply RReLU to the input tensor
        torch::Tensor output;
        if (inplace && input.is_floating_point()) {
            // For inplace operations, we need to make sure the tensor is floating point
            // and we need to clone it to avoid modifying the original
            torch::Tensor input_clone = input.clone();
            output = rrelu(input_clone);
        } else {
            output = rrelu(input);
        }
        
        // Test the functional version as well
        torch::Tensor output_functional = torch::nn::functional::rrelu(
            input, 
            torch::nn::functional::RReLUFuncOptions()
                .lower(lower)
                .upper(upper)
                .inplace(false)
                .training(true)
        );
        
        // Test with different training modes
        rrelu->eval();
        torch::Tensor output_eval = rrelu(input);
        
        rrelu->train();
        torch::Tensor output_train = rrelu(input);
        
        // Test with different device if available
        if (torch::cuda::is_available()) {
            torch::Tensor input_cuda = input.to(torch::kCUDA);
            torch::nn::RReLU rrelu_cuda = torch::nn::RReLU(
                torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(inplace)
            );
            rrelu_cuda->to(torch::kCUDA);
            torch::Tensor output_cuda = rrelu_cuda(input_cuda);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
