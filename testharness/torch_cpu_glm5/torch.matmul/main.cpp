#include "fuzzer_utils.h"
#include <iostream>
#include <stdexcept>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;
        
        // Create two tensors from the fuzz input data.
        // The createTensor function handles parsing dtype, rank, shape, and data.
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // If we have enough data left for at least metadata (2 bytes), create second tensor
        // Otherwise, create a simple 1D tensor to ensure valid API call
        torch::Tensor other_tensor;
        if (offset + 2 <= Size) {
            other_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Fallback: create a small compatible tensor if input exhausted
            other_tensor = torch::randn({1});
        }

        // Perform the matmul operation.
        // We let PyTorch handle validation of dimension compatibility.
        // This explores edge cases like:
        // - Vector x Vector (dot product)
        // - Matrix x Vector
        // - Matrix x Matrix
        // - Batched Matrix x Broadcasted Vector
        // - Batched Matrix x Batched Matrix
        // - Empty tensors (0 dimensions)
        // - Various dtypes (float, double, half, bfloat16, complex, int types, bool)
        auto result = torch::matmul(input_tensor, other_tensor);

        // Consume the result to ensure the operation isn't optimized away
        // and to trigger any lazy evaluation or backend issues.
        // Using sum() is a lightweight way to force computation.
        (void)result.sum().item<float>();

    } catch (const c10::Error &e) {
        // Catch PyTorch-specific errors (C10 errors).
        // These are expected for invalid inputs (e.g., dimension mismatches).
        // We suppress them to keep the fuzzer running.
        // Uncomment for debugging:
        // std::cerr << "C10 error caught: " << e.what() << std::endl;
        return 0;
    } catch (const std::exception &e) {
        // Catch standard exceptions.
        // This includes parsing errors from fuzzer_utils or other unexpected issues.
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // Discard input for libFuzzer
    }

    return 0;
}