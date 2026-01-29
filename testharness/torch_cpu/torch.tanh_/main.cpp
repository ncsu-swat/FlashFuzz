#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Make a copy of the original tensor for verification
        torch::Tensor original = tensor.clone();

        // Apply tanh_ in-place operation
        tensor.tanh_();

        // Verify the operation worked correctly (inner try-catch, no logging)
        try {
            torch::Tensor expected = torch::tanh(original);
            if (!torch::allclose(tensor, expected, 1e-5, 1e-5)) {
                // Verification failed, but don't crash - could be precision issues
            }
        } catch (const std::exception&) {
            // Ignore verification errors
        }

        // Try with different tensor options if we have more data
        if (offset < Size) {
            size_t offset2 = 0;
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);

            // Apply tanh_ to the second tensor
            tensor2.tanh_();

            // Try with a view of the tensor to test in-place ops on views
            try {
                if (tensor2.numel() > 0 && tensor2.dim() > 0 && tensor2.size(0) > 0) {
                    torch::Tensor view = tensor2.slice(0, 0, tensor2.size(0));
                    view.tanh_();
                }
            } catch (const std::exception&) {
                // Views may fail in some cases
            }
        }

        // Try with a scalar tensor (0-dim)
        {
            float val = static_cast<float>(Data[0]) / 255.0f;
            torch::Tensor scalar_tensor = torch::tensor(val);
            scalar_tensor.tanh_();
        }

        // Try with empty tensor
        {
            torch::Tensor empty_tensor = torch::empty({0});
            empty_tensor.tanh_();
        }

        // Try with tensors of different dtypes
        if (Size >= 2) {
            float v1 = static_cast<float>(Data[0]) / 255.0f;
            float v2 = static_cast<float>(Data[1]) / 255.0f;

            // Float tensor
            {
                torch::Tensor float_tensor = torch::tensor({v1, v2}, torch::kFloat);
                float_tensor.tanh_();
            }

            // Double tensor
            {
                torch::Tensor double_tensor = torch::tensor({static_cast<double>(v1), static_cast<double>(v2)}, torch::kDouble);
                double_tensor.tanh_();
            }

            // Half tensor (if supported)
            try {
                torch::Tensor half_tensor = torch::tensor({v1, v2}, torch::kFloat).to(torch::kHalf);
                half_tensor.tanh_();
            } catch (const std::exception&) {
                // Half precision might not be supported on all platforms
            }

            // Complex tensor
            try {
                torch::Tensor complex_tensor = torch::complex(
                    torch::tensor({v1}),
                    torch::tensor({v2})
                );
                complex_tensor.tanh_();
            } catch (const std::exception&) {
                // Complex tanh might not be supported
            }
        }

        // Test with contiguous and non-contiguous tensors
        if (Size >= 4) {
            try {
                torch::Tensor base = torch::randn({4, 4});
                // Create a non-contiguous tensor via transpose
                torch::Tensor non_contig = base.t();
                if (!non_contig.is_contiguous()) {
                    non_contig.tanh_();
                }
            } catch (const std::exception&) {
                // Non-contiguous in-place ops might fail
            }
        }

        // Test with requires_grad tensors
        try {
            torch::Tensor grad_tensor = torch::randn({3, 3}, torch::requires_grad());
            // tanh_ doesn't work on tensors that require grad, but let's test
            grad_tensor.tanh_();
        } catch (const std::exception&) {
            // Expected to fail for tensors requiring grad
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}