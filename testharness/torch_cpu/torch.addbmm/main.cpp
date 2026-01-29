#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <cmath>

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
        // Need enough data for dimensions and scalars
        if (Size < 12) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzzer data (keep them small to avoid OOM)
        uint8_t batch_size = (Data[offset++] % 8) + 1;  // 1-8
        uint8_t n = (Data[offset++] % 16) + 1;          // 1-16
        uint8_t m = (Data[offset++] % 16) + 1;          // 1-16
        uint8_t p = (Data[offset++] % 16) + 1;          // 1-16

        // Extract alpha and beta as floats
        float alpha = 1.0f;
        float beta = 1.0f;
        
        if (offset + 8 <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&beta, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp to reasonable values to avoid numerical issues
            if (!std::isfinite(alpha)) alpha = 1.0f;
            if (!std::isfinite(beta)) beta = 1.0f;
            alpha = std::max(-100.0f, std::min(100.0f, alpha));
            beta = std::max(-100.0f, std::min(100.0f, beta));
        }

        // Determine dtype from fuzzer data
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
            }
        }

        // Create tensors with compatible shapes
        // batch1: (batch_size, n, m)
        // batch2: (batch_size, m, p)
        // input: (n, p)
        torch::Tensor input = torch::randn({n, p}, torch::TensorOptions().dtype(dtype));
        torch::Tensor batch1 = torch::randn({batch_size, n, m}, torch::TensorOptions().dtype(dtype));
        torch::Tensor batch2 = torch::randn({batch_size, m, p}, torch::TensorOptions().dtype(dtype));

        // If we have more data, use it to influence tensor values
        if (offset < Size) {
            torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            // Use fuzz_tensor to modify input in a controlled way
            try {
                if (fuzz_tensor.numel() > 0) {
                    float scale = fuzz_tensor.abs().mean().item<float>();
                    if (std::isfinite(scale) && scale > 0) {
                        input = input * scale;
                    }
                }
            } catch (...) {
                // Ignore errors from fuzz_tensor manipulation
            }
        }

        // Test addbmm with alpha and beta
        try {
            torch::Tensor result = torch::addbmm(input, batch1, batch2, beta, alpha);
            (void)result;
        } catch (const c10::Error &e) {
            // Expected for some input combinations
        }

        // Test with default alpha and beta
        try {
            torch::Tensor result = torch::addbmm(input, batch1, batch2);
            (void)result;
        } catch (const c10::Error &e) {
            // Expected for some input combinations
        }

        // Test addbmm_out
        try {
            torch::Tensor out = torch::empty({n, p}, torch::TensorOptions().dtype(dtype));
            torch::addbmm_out(out, input, batch1, batch2, beta, alpha);
            (void)out;
        } catch (const c10::Error &e) {
            // Expected for some input combinations
        }

        // Test in-place version
        try {
            torch::Tensor inplace = input.clone();
            inplace.addbmm_(batch1, batch2, beta, alpha);
            (void)inplace;
        } catch (const c10::Error &e) {
            // Expected for some input combinations
        }

        // Test with transposed inputs
        try {
            torch::Tensor batch1_t = batch1.transpose(1, 2).contiguous();  // (batch, m, n)
            torch::Tensor batch2_t = batch2.transpose(1, 2).contiguous();  // (batch, p, m)
            torch::Tensor input_t = torch::randn({m, m}, torch::TensorOptions().dtype(dtype));
            // batch1_t @ batch2_t would be (batch, m, m), so input needs to match
            torch::Tensor result = torch::addbmm(input_t, batch1_t, batch2_t, beta, alpha);
            (void)result;
        } catch (const c10::Error &e) {
            // Expected for shape mismatches
        }

        // Test with zero-sized dimension
        try {
            torch::Tensor empty_batch1 = torch::randn({0, n, m}, torch::TensorOptions().dtype(dtype));
            torch::Tensor empty_batch2 = torch::randn({0, m, p}, torch::TensorOptions().dtype(dtype));
            torch::Tensor result = torch::addbmm(input, empty_batch1, empty_batch2, beta, alpha);
            (void)result;
        } catch (const c10::Error &e) {
            // May or may not be supported
        }

        // Test with single batch
        try {
            torch::Tensor single_batch1 = torch::randn({1, n, m}, torch::TensorOptions().dtype(dtype));
            torch::Tensor single_batch2 = torch::randn({1, m, p}, torch::TensorOptions().dtype(dtype));
            torch::Tensor result = torch::addbmm(input, single_batch1, single_batch2, beta, alpha);
            (void)result;
        } catch (const c10::Error &e) {
            // Expected for some combinations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}