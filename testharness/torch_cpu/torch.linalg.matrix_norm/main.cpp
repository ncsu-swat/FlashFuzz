#include "fuzzer_utils.h"
#include <iostream>
#include <ATen/ops/linalg_matrix_norm.h>
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
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - use float for norm computation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already a floating point type
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure tensor has at least 2 dimensions for matrix_norm
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Get remaining bytes for norm parameters
        if (offset >= Size) {
            offset = 0;
        }
        
        // Parse norm type selector
        uint8_t norm_selector = (offset < Size) ? Data[offset++] : 0;
        
        // Parse dim parameter - must be a 2-tuple for matrix_norm
        std::vector<int64_t> dim;
        if (offset < Size) {
            uint8_t dim_selector = Data[offset++];
            if (dim_selector % 3 == 0) {
                dim = {-2, -1};
            } else if (dim_selector % 3 == 1) {
                int64_t last_dim = input.dim() - 1;
                int64_t second_last_dim = std::max(0L, last_dim - 1);
                dim = {second_last_dim, last_dim};
            } else {
                dim = {0, std::min(1L, static_cast<int64_t>(input.dim() - 1))};
            }
        } else {
            dim = {-2, -1};
        }
        
        // Parse keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] % 2 == 0;
        }
        
        // Parse dtype parameter
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            if (dtype_selector % 4 == 1) {
                dtype = torch::kFloat32;
            } else if (dtype_selector % 4 == 2) {
                dtype = torch::kFloat64;
            }
            // else leave as nullopt
        }
        
        torch::Tensor result;
        
        // Call different variants based on norm_selector
        // Use at::linalg_matrix_norm which is the C++ ATen function
        try {
            switch (norm_selector % 6) {
                case 0:
                    // Frobenius norm (string version)
                    result = at::linalg_matrix_norm(input, "fro", dim, keepdim, dtype);
                    break;
                case 1:
                    // Nuclear norm (string version)
                    result = at::linalg_matrix_norm(input, "nuc", dim, keepdim, dtype);
                    break;
                case 2:
                    // 1-norm (numeric)
                    result = at::linalg_matrix_norm(input, 1.0, dim, keepdim, dtype);
                    break;
                case 3:
                    // 2-norm (numeric)
                    result = at::linalg_matrix_norm(input, 2.0, dim, keepdim, dtype);
                    break;
                case 4:
                    // Inf-norm (numeric)
                    result = at::linalg_matrix_norm(input, INFINITY, dim, keepdim, dtype);
                    break;
                case 5:
                    // -Inf-norm (numeric)
                    result = at::linalg_matrix_norm(input, -INFINITY, dim, keepdim, dtype);
                    break;
            }
        } catch (const c10::Error&) {
            // Expected failures for invalid inputs (e.g., shape mismatches)
            return 0;
        } catch (const std::runtime_error&) {
            // Expected failures
            return 0;
        }
        
        // Basic validation
        if (result.defined() && result.numel() > 0) {
            // Access the result to ensure computation completed
            (void)result.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}