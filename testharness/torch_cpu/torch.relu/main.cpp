#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // 1. Construct a tensor from the fuzz data.
        // This function handles parsing random bytes into tensor metadata (rank, shape, dtype)
        // and filling the tensor data. It throws if the data is insufficient for metadata.
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // 2. Test the functional API: torch::relu(input)
        // This is the standard out-of-place operation.
        torch::Tensor output = torch::relu(input);

        // 3. Test the method API: input.relu()
        // This ensures the Tensor class method binding functions correctly.
        torch::Tensor output_method = input.relu();

        // 4. Test the in-place API: torch::relu_(input)
        // We clone the input first to ensure we are operating on a distinct memory block
        // and to test the specific code path for in-place mutation without affecting previous variables.
        // Note: Some dtypes (e.g. Complex) might throw "not implemented" exceptions, which are caught below.
        torch::Tensor input_inplace = input.clone();
        torch::relu_(input_inplace);

    }
    catch (const std::exception &e)
    {
        // Catch exceptions such as "unimplemented data type" or invalid shapes
        // that are expected during fuzzing of random inputs.
        std::cout << "Exception caught: " << e.what() << std::endl; 
        return -1; // discard the input for libfuzzer
    }
    return 0; // keep the input
}