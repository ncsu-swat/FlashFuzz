#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Create input tensor from fuzz data
        // This utility handles random shapes, ranks, and dtypes based on the input bytes.
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // 1. Standard Functional Call
        // sigmoid(input) -> Tensor
        torch::Tensor output = torch::sigmoid(input);

        // 2. Explore variations if data remains
        if (offset < Size)
        {
            uint8_t selector = Data[offset++] % 2;
            
            if (selector == 0)
            {
                // Test in-place: input.sigmoid_()
                // Sigmoid maps Int -> Float, so in-place is only valid if input is already Float/Complex.
                if (torch::isFloatingType(input.scalar_type()) || torch::isComplexType(input.scalar_type()))
                {
                    // Clone input to ensure we have a fresh mutable tensor
                    torch::Tensor input_mutable = input.clone();
                    input_mutable.sigmoid_();
                }
            }
            else
            {
                // Test out= argument: sigmoid(input, *, out=out_tensor)
                // Use empty_like based on the calculated output to ensure compatible shape/dtype.
                // This minimizes "invalid argument" errors regarding mismatched dtypes for the 'out' parameter.
                torch::Tensor out = torch::empty_like(output);
                torch::sigmoid_out(out, input);
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input for libfuzzer
    }
    return 0; // keep the input
}