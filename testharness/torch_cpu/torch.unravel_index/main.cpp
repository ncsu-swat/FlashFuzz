#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <limits>

// Manual implementation of unravel_index since it's not in C++ API
std::vector<torch::Tensor> manual_unravel_index(const torch::Tensor& indices, c10::IntArrayRef shape) {
    // Validate inputs
    if (shape.empty()) {
        throw std::runtime_error("shape cannot be empty");
    }
    
    // Compute strides (row-major order)
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    
    // Convert indices to int64
    torch::Tensor flat_indices = indices.to(torch::kInt64).reshape({-1});
    
    // Compute coordinate for each dimension
    std::vector<torch::Tensor> result;
    torch::Tensor remaining = flat_indices.clone();
    
    for (size_t i = 0; i < shape.size(); ++i) {
        torch::Tensor coord = remaining.div(strides[i], "trunc").to(torch::kInt64);
        remaining = remaining - coord * strides[i];
        result.push_back(coord);
    }
    
    return result;
}

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor for indices
        torch::Tensor indices = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure indices are integers
        if (!indices.is_floating_point()) {
            if (indices.scalar_type() != torch::kInt64 && indices.scalar_type() != torch::kInt32) {
                indices = indices.to(torch::kInt64);
            }
        } else {
            indices = indices.to(torch::kInt64);
        }
        
        // Make indices non-negative for valid testing
        indices = indices.abs();
        
        // Parse dimensions
        std::vector<int64_t> dims;
        const size_t max_dims = 5;
        
        // Determine number of dimensions to use
        uint8_t num_dims = 1;
        if (offset < Size) {
            num_dims = (Data[offset++] % max_dims) + 1;
        }
        
        // Parse each dimension
        for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
            int64_t dim_value;
            std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure dimension is positive
            if (dim_value <= 0) {
                dim_value = (-dim_value) % 100 + 1;
            } else {
                dim_value = dim_value % 100 + 1;
            }
            
            dims.push_back(dim_value);
        }
        
        // If we couldn't parse any dimensions, use a default
        if (dims.empty()) {
            dims.push_back(10);
        }
        
        // Calculate product of dimensions for index clamping
        int64_t prod = 1;
        for (auto d : dims) {
            if (d > 0 && prod < std::numeric_limits<int64_t>::max() / d) {
                prod *= d;
            }
        }
        
        // Variant selector
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 8;
        }
        
        // Variant 1: Basic usage with 1D indices
        if (variant == 0 || variant == 1) {
            torch::Tensor flat_indices = indices.reshape({-1});
            // Clamp indices to valid range
            if (prod > 0) {
                flat_indices = flat_indices % prod;
            }
            auto result = manual_unravel_index(flat_indices, c10::IntArrayRef(dims));
            // Result is a vector of tensors, one for each dimension
            (void)result;
        }
        
        // Variant 2: Scalar index
        if (variant == 2) {
            torch::Tensor scalar_idx = torch::tensor({0}, torch::kInt64);
            if (indices.numel() > 0) {
                int64_t val = indices.reshape({-1})[0].item<int64_t>();
                if (prod > 0) {
                    val = val % prod;
                }
                scalar_idx = torch::tensor({val}, torch::kInt64);
            }
            auto result = manual_unravel_index(scalar_idx, c10::IntArrayRef(dims));
            (void)result;
        }
        
        // Variant 3: With int32 indices
        if (variant == 3) {
            torch::Tensor int32_indices = indices.reshape({-1}).to(torch::kInt32);
            if (prod > 0) {
                int32_indices = int32_indices % static_cast<int32_t>(std::min(prod, static_cast<int64_t>(INT32_MAX)));
            }
            auto result = manual_unravel_index(int32_indices, c10::IntArrayRef(dims));
            (void)result;
        }
        
        // Variant 4: Multi-dimensional input indices (will be flattened internally)
        if (variant == 4) {
            torch::Tensor flat = indices.reshape({-1});
            if (prod > 0) {
                flat = flat % prod;
            }
            // Reshape to 2D
            int64_t numel = flat.numel();
            if (numel >= 2) {
                int64_t rows = 2;
                int64_t cols = numel / 2;
                torch::Tensor reshaped = flat.narrow(0, 0, rows * cols).reshape({rows, cols});
                auto result = manual_unravel_index(reshaped, c10::IntArrayRef(dims));
                (void)result;
            }
        }
        
        // Variant 5: Edge case with empty tensor
        if (variant == 5) {
            torch::Tensor empty_indices = torch::empty({0}, torch::kInt64);
            auto result = manual_unravel_index(empty_indices, c10::IntArrayRef(dims));
            (void)result;
        }
        
        // Variant 6: Edge case - test behavior with out-of-bounds indices
        if (variant == 6) {
            if (prod > 0) {
                torch::Tensor oob_indices = indices.reshape({-1}).abs() + prod;
                try {
                    // This may produce incorrect results (not throw) since we're using manual impl
                    auto result = manual_unravel_index(oob_indices, c10::IntArrayRef(dims));
                    (void)result;
                } catch (...) {
                    // Silently catch - expected behavior
                }
            }
        }
        
        // Variant 7: Large number of dimensions
        if (variant == 7) {
            std::vector<int64_t> many_dims;
            for (int i = 0; i < 4; i++) {
                many_dims.push_back(2 + (i % 3));
            }
            int64_t many_prod = 1;
            for (auto d : many_dims) {
                many_prod *= d;
            }
            torch::Tensor clamped = indices.reshape({-1}).abs() % many_prod;
            auto result = manual_unravel_index(clamped, c10::IntArrayRef(many_dims));
            (void)result;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}