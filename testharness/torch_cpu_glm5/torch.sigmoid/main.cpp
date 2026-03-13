#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        size_t offset = 0;

        // Create input tensor with varied ranks, shapes, and dtypes
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Call torch::sigmoid
        torch::Tensor output_tensor = torch::sigmoid(input_tensor);

        // Exercise the output to ensure it's valid (accessing data)
        if (output_tensor.numel() > 0) {
            // Force computation of output values to catch potential NaN/Inf issues
            volatile auto first_val = output_tensor.data_ptr<float>()[0];
            (void)first_val;
        }

    } catch (const c10::Error &e) {
        // Catch PyTorch-specific errors and continue fuzzing
        std::cerr << "c10::Error caught: " << e.what() << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}