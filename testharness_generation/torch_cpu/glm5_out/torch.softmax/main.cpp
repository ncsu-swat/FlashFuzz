#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        if (Size < 3) {
            return 0;
        }

        size_t offset = 0;

        // 1. Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // 2. Determine dim argument
        // Use next byte to select dim. We explore valid and potentially invalid dims.
        int64_t dim = 0;
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            int64_t rank = input_tensor.dim();

            if (rank == 0) {
                // For scalar tensors, dim must be 0 (or -1 which wraps to 0, but 0 is canonical)
                dim = 0;
            } else {
                // Map byte to range [-rank, rank - 1] to test valid dims.
                // Also allow testing slightly out of bounds to check error handling (though we catch exceptions).
                // Let's bias towards valid dims.
                int64_t range = 2 * rank;
                int64_t dim_val = static_cast<int64_t>(dim_byte % (range + 4)); // +4 to allow some out-of-bounds
                dim = dim_val - rank; // maps to [-rank, rank+3]
            }
        } else {
            // Default if no data left
            dim = input_tensor.dim() > 0 ? 0 : 0;
        }

        // 3. Determine dtype (optional)
        // Use next byte to decide if we specify dtype
        torch::Dtype dtype = torch::kFloat; // Default
        bool has_dtype = false;
        if (offset < Size) {
            uint8_t dtype_flag = Data[offset++];
            if (dtype_flag % 3 == 0) {
                has_dtype = true;
                // Select a valid softmax dtype (float, double, half, bfloat16)
                uint8_t type_sel = dtype_flag >> 2;
                switch (type_sel % 4) {
                    case 0: dtype = torch::kFloat; break;
                    case 1: dtype = torch::kDouble; break;
                    case 2: dtype = torch::kHalf; break;
                    case 3: dtype = torch::kBFloat16; break;
                }
            }
        }

        // 4. Call torch::softmax
        // API: softmax(input, dim, *, dtype=None)
        // Note: c10::optional<torch::Dtype> for the dtype argument
        c10::optional<torch::Dtype> dtype_opt;
        if (has_dtype) {
            dtype_opt = dtype;
        }

        // Softmax requires float-like inputs typically, but we let the API handle type errors.
        // We catch exceptions to keep the fuzzer running.
        torch::Tensor output = torch::softmax(input_tensor, dim, dtype_opt);

        // Basic sanity check (optional, but good for detecting crashes)
        if (output.defined()) {
            // Check if output shape matches input shape (softmax preserves shape)
            if (input_tensor.sizes() != output.sizes()) {
                // This indicates a logic error in the op or our understanding
                std::cerr << "Shape mismatch: input " << input_tensor.sizes() 
                          << " vs output " << output.sizes() << std::endl;
            }
        }

    } catch (const c10::Error &e) {
        // Catch PyTorch errors specifically (e.g., dim out of range, dtype errors)
        // We expect these for fuzz inputs, so we just continue.
        // std::cout << "c10::Error caught: " << e.what() << std::endl;
        return 0;
    } catch (const std::exception &e) {
        // Catch other standard exceptions
        std::cout << "Exception caught: " << e.what() << std::endl;
        return 0;
    }

    return 0;
}