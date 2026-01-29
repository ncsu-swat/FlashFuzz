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
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzzer data for controlled matrix sizes
        // K = rows, N = columns
        int64_t K = static_cast<int64_t>((Data[offset++] % 64) + 1);  // 1-64
        int64_t N = static_cast<int64_t>((Data[offset++] % 64) + 1);  // 1-64
        
        // FBGEMM requires N to be a multiple of certain alignment (typically 4 or 8)
        // Round N up to multiple of 4 for better compatibility
        N = ((N + 3) / 4) * 4;
        
        // Determine which variant to test
        bool use_two_arg_variant = (Size > offset) && (Data[offset++] % 2 == 0);

        // Create a 2D tensor with Int8 dtype (required by fbgemm)
        // Use remaining data to fill tensor values
        size_t remaining = Size - offset;
        int64_t total_elements = K * N;
        
        std::vector<int8_t> tensor_data(total_elements);
        for (int64_t i = 0; i < total_elements; ++i) {
            if (i < static_cast<int64_t>(remaining)) {
                tensor_data[i] = static_cast<int8_t>(Data[offset + i]);
            } else {
                // Fill remaining with pattern derived from available data
                tensor_data[i] = static_cast<int8_t>(i % 256);
            }
        }

        torch::Tensor input_tensor = torch::from_blob(
            tensor_data.data(),
            {K, N},
            torch::kInt8
        ).clone();  // Clone to own the memory

        // Ensure tensor is contiguous (required by FBGEMM)
        input_tensor = input_tensor.contiguous();

        torch::Tensor packed_weights;
        
        try {
            if (use_two_arg_variant) {
                // Two-argument version: infers K and N from tensor shape
                packed_weights = torch::fbgemm_pack_quantized_matrix(input_tensor);
            } else {
                // Four-argument version: explicitly specifies K and N
                packed_weights = torch::fbgemm_pack_quantized_matrix(
                    input_tensor,
                    K,
                    N
                );
            }
        } catch (const c10::Error& e) {
            // FBGEMM may not be available on all platforms (e.g., non-x86)
            // or may fail for certain dimension combinations
            // Silently catch these expected failures
            return 0;
        }

        // Verify the packed result is defined
        if (packed_weights.defined()) {
            // Just check basic properties - the packed format is opaque
            auto sizes = packed_weights.sizes();
            auto numel = packed_weights.numel();
            (void)sizes;
            (void)numel;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}