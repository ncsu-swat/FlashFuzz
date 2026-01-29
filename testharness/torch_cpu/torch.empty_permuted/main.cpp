#include "fuzzer_utils.h"
#include <iostream>
#include <algorithm>
#include <numeric>

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
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Determine number of dimensions (1-6)
        uint8_t ndim = (Data[offset++] % 6) + 1;

        // Generate shape from fuzzer data
        std::vector<int64_t> shape;
        for (int i = 0; i < ndim && offset < Size; ++i) {
            // Use small dimensions to avoid OOM (1-16)
            int64_t dim_size = (Data[offset++] % 16) + 1;
            shape.push_back(dim_size);
        }

        // Ensure we have enough dimensions
        while (shape.size() < static_cast<size_t>(ndim)) {
            shape.push_back(1);
        }

        // Generate a valid permutation using Fisher-Yates shuffle
        std::vector<int64_t> physical_layout(ndim);
        std::iota(physical_layout.begin(), physical_layout.end(), 0);

        // Shuffle based on remaining fuzzer data
        for (int64_t i = ndim - 1; i > 0 && offset < Size; --i) {
            int64_t j = Data[offset++] % (i + 1);
            std::swap(physical_layout[i], physical_layout[j]);
        }

        // Determine dtype from fuzzer data
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kInt; break;
                case 3: dtype = torch::kLong; break;
            }
        }

        auto options = torch::TensorOptions().dtype(dtype);

        // Call empty_permuted - creates tensor with specified shape and memory layout
        // physical_layout[i] specifies which logical dimension is stored in physical dimension i
        torch::Tensor result = torch::empty_permuted(shape, physical_layout, options);

        // Verify the result has the expected logical shape
        if (result.dim() != ndim) {
            throw std::runtime_error("Unexpected number of dimensions");
        }

        for (int64_t i = 0; i < ndim; ++i) {
            if (result.size(i) != shape[i]) {
                throw std::runtime_error("Unexpected shape");
            }
        }

        // Test that the tensor is usable - fill with a value
        if (result.numel() > 0 && result.numel() < 10000) {
            try {
                result.fill_(1.0);
                
                // Test contiguous conversion
                torch::Tensor contiguous = result.contiguous();
                
                // Test basic operations on the permuted tensor
                if (dtype == torch::kFloat || dtype == torch::kDouble) {
                    torch::Tensor sum = result.sum();
                }
            }
            catch (const std::exception &) {
                // Expected failures for some operations - silently ignore
            }
        }

        // Test empty_permuted with different options combinations
        if (offset < Size && (Data[offset] & 1)) {
            try {
                auto options2 = torch::TensorOptions()
                    .dtype(dtype)
                    .requires_grad(dtype == torch::kFloat || dtype == torch::kDouble);
                torch::Tensor result2 = torch::empty_permuted(shape, physical_layout, options2);
            }
            catch (const std::exception &) {
                // Silently ignore expected failures
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