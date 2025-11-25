#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate the 'input' tensor using fuzzer utilities.
        // This handles consumption of bytes for dtype, rank, shape, and tensor data.
        // If there isn't enough data for valid metadata, it throws an exception which is caught below.
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // Generate the 'other' tensor.
        torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);

        // Decide whether to use the optional 'out' parameter.
        // This allows the fuzzer to explore paths where an output tensor is provided,
        // including resizing logic and dtype checks.
        bool use_out = false;
        if (offset < Size) {
            // Consume a byte to decide. 
            use_out = (Data[offset++] % 2 != 0);
        }

        if (use_out) {
            // Generate a tensor to serve as the explicit output buffer.
            // The shape and dtype of 'out' may not match the result of matmul;
            // PyTorch handles resizing or throws if incompatible.
            torch::Tensor out = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Invoke the in-place/out variant
            // Note: The documentation states 1D dot product does not support 'out'.
            // If input/other are both 1D, this is expected to throw.
            torch::matmul_out(out, input, other);
        } else {
            // Invoke the standard variant
            torch::Tensor result = torch::matmul(input, other);
            
            // Access result properties to ensure the object is valid
            if (result.defined()) {
                volatile int64_t dim = result.dim();
                (void)dim; 
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