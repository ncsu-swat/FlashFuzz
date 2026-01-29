#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        // Need enough bytes for tensor creation and parameters
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract matrix dimensions from fuzzer data
        // PCA lowrank requires a 2D input tensor (matrix)
        uint8_t rows_byte = Data[offset++];
        uint8_t cols_byte = Data[offset++];
        
        // Ensure reasonable dimensions (at least 2x2, at most 64x64 for performance)
        int64_t rows = (rows_byte % 63) + 2;  // 2-64
        int64_t cols = (cols_byte % 63) + 2;  // 2-64

        // Extract parameters
        uint8_t q_byte = Data[offset++];
        int64_t min_dim = std::min(rows, cols);
        int64_t q = (q_byte % (min_dim - 1)) + 1;  // q must be in [1, min(rows, cols)-1]

        bool center = Data[offset++] & 0x1;

        // Create a 2D tensor for PCA
        torch::Tensor input;
        if (offset < Size) {
            // Use remaining data to seed tensor values
            size_t remaining = Size - offset;
            std::vector<float> values(rows * cols);
            for (int64_t i = 0; i < rows * cols; i++) {
                values[i] = static_cast<float>(Data[offset + (i % remaining)]) / 255.0f - 0.5f;
            }
            input = torch::from_blob(values.data(), {rows, cols}, torch::kFloat32).clone();
        } else {
            input = torch::randn({rows, cols});
        }

        // Ensure input is float type and requires grad for better coverage
        input = input.to(torch::kFloat32);

        // Call torch::pca_lowrank
        try {
            auto result = torch::pca_lowrank(input, q, center);

            // Unpack the result (U, S, V)
            auto U = std::get<0>(result);
            auto S = std::get<1>(result);
            auto V = std::get<2>(result);

            // Basic sanity checks on output shapes
            auto u_sum = U.sum();
            auto s_sum = S.sum();
            auto v_sum = V.sum();

            // Try reconstruction to exercise more code paths
            auto reconstructed = torch::matmul(U, torch::matmul(torch::diag(S), V.t()));
            auto reconstruction_error = torch::norm(reconstructed - input);

        } catch (const c10::Error& e) {
            // Expected failures due to invalid inputs - silently ignore
        } catch (const std::runtime_error& e) {
            // Expected failures - silently ignore
        }

        // Try with different q values if we have more data
        if (offset + 1 < Size) {
            uint8_t q2_byte = Data[offset++];
            int64_t q2 = (q2_byte % (min_dim - 1)) + 1;
            bool center2 = (offset < Size) ? (Data[offset++] & 0x1) : !center;

            try {
                auto result2 = torch::pca_lowrank(input, q2, center2);
                auto U2 = std::get<0>(result2);
                auto S2 = std::get<1>(result2);
            } catch (const c10::Error& e) {
                // Expected failures - silently ignore
            } catch (const std::runtime_error& e) {
                // Expected failures - silently ignore
            }
        }

        // Test with batched input for additional coverage
        if (offset + 1 < Size) {
            uint8_t batch_byte = Data[offset++];
            int64_t batch_size = (batch_byte % 3) + 1;  // 1-3 batches
            
            // Create smaller batched input
            int64_t small_rows = std::min(rows, (int64_t)16);
            int64_t small_cols = std::min(cols, (int64_t)16);
            int64_t small_q = std::min(q, std::min(small_rows, small_cols) - 1);
            if (small_q < 1) small_q = 1;
            
            torch::Tensor batched_input = torch::randn({batch_size, small_rows, small_cols});

            try {
                auto batched_result = torch::pca_lowrank(batched_input, small_q, center);
                auto U_batch = std::get<0>(batched_result);
            } catch (const c10::Error& e) {
                // Expected failures - silently ignore
            } catch (const std::runtime_error& e) {
                // Expected failures - silently ignore
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