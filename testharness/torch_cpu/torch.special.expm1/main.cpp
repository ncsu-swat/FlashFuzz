#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // Apply torch.special.expm1 operation (computes exp(x) - 1)
        torch::Tensor result = torch::special::expm1(input);

        // Verify result is defined and has correct shape
        if (!result.defined() || result.sizes() != input.sizes()) {
            return 0;
        }

        // Force computation
        result.sum().item<float>();

        // Test with out parameter variant
        try {
            torch::Tensor output = torch::empty_like(input);
            torch::special::expm1_out(output, input);
            output.sum().item<float>();
        } catch (...) {
            // Silently ignore expected failures
        }

        // Test with different dtypes based on fuzzer data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset % Size];
            
            try {
                if (dtype_selector % 3 == 0) {
                    // Test with float32
                    torch::Tensor float_input = input.to(torch::kFloat32);
                    torch::Tensor float_result = torch::special::expm1(float_input);
                    float_result.sum().item<float>();
                } else if (dtype_selector % 3 == 1) {
                    // Test with float64
                    torch::Tensor double_input = input.to(torch::kFloat64);
                    torch::Tensor double_result = torch::special::expm1(double_input);
                    double_result.sum().item<double>();
                } else {
                    // Test in-place style using out variant with pre-allocated tensor
                    torch::Tensor out_tensor = torch::zeros_like(input);
                    torch::special::expm1_out(out_tensor, input);
                    out_tensor.sum().item<float>();
                }
            } catch (...) {
                // Silently ignore dtype conversion failures
            }
        }

        // Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2) {
            try {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor trans_result = torch::special::expm1(transposed);
                trans_result.sum().item<float>();
            } catch (...) {
                // Silently ignore failures
            }
        }

        // Test with sliced tensor (non-contiguous)
        if (input.numel() > 2) {
            try {
                torch::Tensor sliced = input.flatten().slice(0, 0, input.numel() / 2);
                torch::Tensor slice_result = torch::special::expm1(sliced);
                slice_result.sum().item<float>();
            } catch (...) {
                // Silently ignore failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}