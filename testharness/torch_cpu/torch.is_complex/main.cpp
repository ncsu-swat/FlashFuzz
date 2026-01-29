#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 2) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Test is_complex on the original tensor
        bool result = torch::is_complex(tensor);

        // Use the result to prevent optimization
        volatile bool use_result = result;
        (void)use_result;

        // Also test with explicitly created complex tensors to ensure coverage
        // Use remaining data to decide which additional tests to run
        if (offset < Size) {
            uint8_t test_selector = Data[offset++] % 4;

            torch::Tensor test_tensor;
            switch (test_selector) {
                case 0:
                    // Test with ComplexFloat tensor
                    test_tensor = torch::zeros({2, 2}, torch::kComplexFloat);
                    break;
                case 1:
                    // Test with ComplexDouble tensor
                    test_tensor = torch::zeros({2, 2}, torch::kComplexDouble);
                    break;
                case 2:
                    // Test with real Float tensor
                    test_tensor = torch::zeros({2, 2}, torch::kFloat);
                    break;
                case 3:
                    // Test with real Double tensor
                    test_tensor = torch::zeros({2, 2}, torch::kDouble);
                    break;
            }

            bool explicit_result = torch::is_complex(test_tensor);
            volatile bool use_explicit = explicit_result;
            (void)use_explicit;
        }

        // Test on a clone to cover different tensor states
        torch::Tensor clone = tensor.clone();
        bool clone_result = torch::is_complex(clone);
        volatile bool use_clone = clone_result;
        (void)use_clone;

        // Test on contiguous version
        torch::Tensor contiguous = tensor.contiguous();
        bool contiguous_result = torch::is_complex(contiguous);
        volatile bool use_contiguous = contiguous_result;
        (void)use_contiguous;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}