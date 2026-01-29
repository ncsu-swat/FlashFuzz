#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
            std::memcpy(&lower, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize lower bound
            if (!std::isfinite(lower)) {
                lower = 0.125;
            }
            // Clamp to reasonable range [0, 1]
            lower = std::fabs(lower);
            if (lower > 1.0) {
                lower = std::fmod(lower, 1.0);
            }
        }
        
        if (offset + 8 <= Size) {
            std::memcpy(&upper, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Sanitize upper bound
            if (!std::isfinite(upper)) {
                upper = 0.3333333333333333;
            }
            // Clamp to reasonable range [0, 1]
            upper = std::fabs(upper);
            if (upper > 1.0) {
                upper = std::fmod(upper, 1.0);
            }
        }
        
        // Ensure lower <= upper
        if (lower > upper) {
            std::swap(lower, upper);
        }
        
        if (offset < Size) {
            inplace = Data[offset] & 0x1;
            offset++;
        }
        
        // Create RReLU module
        torch::nn::RReLU rrelu(
            torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(false)
        );
        
        // Apply RReLU to the input tensor (non-inplace first)
        torch::Tensor output = rrelu(input);
        
        // Test inplace operation only with floating point tensors
        if (inplace && input.is_floating_point()) {
            try {
                torch::nn::RReLU rrelu_inplace(
                    torch::nn::RReLUOptions().lower(lower).upper(upper).inplace(true)
                );
                torch::Tensor input_clone = input.clone();
                torch::Tensor output_inplace = rrelu_inplace(input_clone);
            } catch (const std::exception &) {
                // Inplace operations may fail for various reasons - expected
            }
        }
        
        // Test the functional version
        try {
            torch::Tensor output_functional = torch::nn::functional::rrelu(
                input, 
                torch::nn::functional::RReLUFuncOptions()
                    .lower(lower)
                    .upper(upper)
                    .inplace(false)
                    .training(true)
            );
        } catch (const std::exception &) {
            // Expected for certain inputs
        }
        
        // Test with different training modes
        rrelu->eval();
        torch::Tensor output_eval = rrelu(input);
        
        rrelu->train();
        torch::Tensor output_train = rrelu(input);
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}