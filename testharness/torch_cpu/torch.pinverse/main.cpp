#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some data for tensor dimensions and values
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Read matrix dimensions from fuzzer data
        uint8_t rows_raw = Data[offset++];
        uint8_t cols_raw = Data[offset++];
        
        // Constrain dimensions to reasonable sizes (1-32) to avoid OOM
        int64_t rows = (rows_raw % 32) + 1;
        int64_t cols = (cols_raw % 32) + 1;

        // Calculate how many float values we need
        size_t num_elements = static_cast<size_t>(rows * cols);
        size_t required_bytes = num_elements * sizeof(float);
        
        if (offset + required_bytes > Size) {
            // Not enough data, use what we have
            rows = 2;
            cols = 2;
            num_elements = 4;
            required_bytes = num_elements * sizeof(float);
            if (offset + required_bytes > Size) {
                return 0;
            }
        }

        // Create tensor from fuzzer data
        std::vector<float> values(num_elements);
        for (size_t i = 0; i < num_elements && offset + sizeof(float) <= Size; i++) {
            std::memcpy(&values[i], Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize values to avoid NaN/Inf issues in matrix operations
            if (!std::isfinite(values[i])) {
                values[i] = 1.0f;
            }
        }

        torch::Tensor input = torch::from_blob(values.data(), {rows, cols}, torch::kFloat32).clone();

        // Test 1: Basic pinverse with default rcond
        try {
            torch::Tensor result = torch::pinverse(input);
            (void)result;
        } catch (const std::exception &) {
            // Shape/computation errors are expected for some inputs
        }

        // Test 2: pinverse with custom rcond from fuzzer data
        if (offset + sizeof(float) <= Size) {
            float rcond_raw;
            std::memcpy(&rcond_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize rcond to a reasonable positive value
            double rcond = std::abs(static_cast<double>(rcond_raw));
            if (!std::isfinite(rcond)) {
                rcond = 1e-7;
            }
            rcond = std::min(rcond, 1.0);  // Cap at 1.0
            
            try {
                torch::Tensor result = torch::pinverse(input, rcond);
                (void)result;
            } catch (const std::exception &) {
                // Expected for some inputs
            }
        }

        // Test 3: Various rcond edge cases with inner try-catch
        // Very small rcond
        try {
            torch::Tensor result = torch::pinverse(input, 1.0e-15);
            (void)result;
        } catch (const std::exception &) {
            // Expected
        }

        // Zero rcond
        try {
            torch::Tensor result = torch::pinverse(input, 0.0);
            (void)result;
        } catch (const std::exception &) {
            // Expected
        }

        // Test 4: Symmetric matrices (pinverse behaves differently)
        if (rows == cols) {
            try {
                // Make input symmetric: A = (A + A^T) / 2
                torch::Tensor symmetric = (input + input.transpose(0, 1)) / 2.0;
                torch::Tensor result = torch::pinverse(symmetric);
                (void)result;
            } catch (const std::exception &) {
                // Expected for some matrices
            }
        }

        // Test 5: Batched pinverse (3D tensor)
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t batch_raw = Data[offset++];
            int64_t batch_size = (batch_raw % 4) + 1;  // 1-4 batches
            
            try {
                torch::Tensor batched = input.unsqueeze(0).expand({batch_size, rows, cols}).clone();
                torch::Tensor result = torch::pinverse(batched);
                (void)result;
            } catch (const std::exception &) {
                // Expected for some configurations
            }
        }

        // Test 6: Different dtypes
        try {
            torch::Tensor input_double = input.to(torch::kFloat64);
            torch::Tensor result = torch::pinverse(input_double);
            (void)result;
        } catch (const std::exception &) {
            // Expected
        }

        // Test 7: Tall matrix (more rows than columns)
        try {
            int64_t tall_rows = std::max(rows, cols) + 1;
            int64_t tall_cols = std::min(rows, cols);
            torch::Tensor tall = torch::randn({tall_rows, tall_cols});
            torch::Tensor result = torch::pinverse(tall);
            (void)result;
        } catch (const std::exception &) {
            // Expected
        }

        // Test 8: Wide matrix (more columns than rows)
        try {
            int64_t wide_rows = std::min(rows, cols);
            int64_t wide_cols = std::max(rows, cols) + 1;
            torch::Tensor wide = torch::randn({wide_rows, wide_cols});
            torch::Tensor result = torch::pinverse(wide);
            (void)result;
        } catch (const std::exception &) {
            // Expected
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}