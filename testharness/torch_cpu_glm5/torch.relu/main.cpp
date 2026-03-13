#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        size_t offset = 0;

        // Need at least 2 bytes for basic tensor metadata
        if (Size < 2) {
            return 0;
        }

        // Create input tensor using fuzzer utilities
        // This explores various ranks (0-4), dtypes, and shapes
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Test torch::relu with the constructed tensor
        // ReLU: f(x) = max(0, x)
        torch::Tensor result = torch::relu(input_tensor);

        // Basic sanity check: result should have same shape as input
        if (!result.sizes().empty() && !input_tensor.sizes().empty()) {
            if (result.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch after relu" << std::endl;
            }
        }

        // Test with in-place variant if tensor is suitable
        // Clone to avoid modifying the original for potential further testing
        if (input_tensor.numel() > 0 && input_tensor.numel() < 1000) {
            torch::Tensor in_place_tensor = input_tensor.clone();
            try {
                in_place_tensor.relu_();
            } catch (const c10::Error &e) {
                // Some dtypes may not support in-place operations
                // Catch narrowly and continue
            }
        }

        // Test functional variant with explicit output tensor
        if (input_tensor.numel() > 0) {
            try {
                torch::Tensor output_tensor = torch::empty_like(input_tensor);
                torch::relu_out(output_tensor, input_tensor);
            } catch (const c10::Error &e) {
                // Some dtype combinations may not be supported for out variant
            }
        }

    } catch (const c10::Error &e) {
        // Catch PyTorch-specific errors and continue fuzzing
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}