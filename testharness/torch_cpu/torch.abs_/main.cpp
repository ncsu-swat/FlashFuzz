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
        // Skip inputs that are too small
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Apply the abs_ operation in-place
        tensor.abs_();

        // Test with different dtypes based on fuzzer input
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::Tensor typed_tensor;
            
            try {
                switch (dtype_selector) {
                    case 0:
                        typed_tensor = tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = tensor.to(torch::kInt32);
                        break;
                    default:
                        typed_tensor = tensor.to(torch::kInt64);
                        break;
                }
                typed_tensor.abs_();
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }

        // Test on a view of the tensor if possible
        if (tensor.numel() > 1 && tensor.dim() > 0) {
            try {
                auto view = tensor.slice(0, 0, std::max<int64_t>(1, tensor.size(0) / 2));
                view.abs_();
            } catch (...) {
                // Silently ignore view operation failures
            }
        }

        // Test with contiguous tensor
        try {
            torch::Tensor contig = tensor.contiguous();
            contig.abs_();
        } catch (...) {
            // Silently ignore
        }

        // Test with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            empty_tensor.abs_();
        } catch (...) {
            // Silently ignore
        }

        // Test with scalar tensor
        if (offset < Size) {
            try {
                float val = static_cast<float>(static_cast<int8_t>(Data[offset]));
                torch::Tensor scalar_tensor = torch::tensor(val);
                scalar_tensor.abs_();
            } catch (...) {
                // Silently ignore
            }
        }

        // Test with negative values explicitly
        try {
            torch::Tensor neg_tensor = torch::randn({4, 4}) * -1.0;
            neg_tensor.abs_();
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}