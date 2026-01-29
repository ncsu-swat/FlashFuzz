#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Read shape parameters from fuzzer data
        uint8_t batch1 = (Data[offset++] % 5) + 1;  // 1-5
        uint8_t batch2 = (Data[offset++] % 5) + 1;  // 1-5
        uint8_t feat_dim = (Data[offset++] % 8) + 1; // 1-8
        uint8_t extra_batch = Data[offset++] % 3;    // 0-2 for optional batch dimension
        
        // Create two input tensors for cdist
        // cdist requires at least 2D tensors: (P, M) and (R, M) or batched (B, P, M) and (B, R, M)
        torch::Tensor x1, x2;
        
        if (extra_batch > 0) {
            // Batched version
            x1 = torch::rand({extra_batch, batch1, feat_dim});
            x2 = torch::rand({extra_batch, batch2, feat_dim});
        } else {
            // Non-batched version
            x1 = torch::rand({batch1, feat_dim});
            x2 = torch::rand({batch2, feat_dim});
        }
        
        // Parse p-norm value from remaining data
        double p = 2.0; // Default p-norm (Euclidean)
        if (offset + sizeof(float) <= Size) {
            float p_raw;
            std::memcpy(&p_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Sanitize p value - cdist requires p >= 0
            if (!std::isnan(p_raw) && !std::isinf(p_raw) && p_raw >= 0.0f) {
                p = static_cast<double>(p_raw);
                // Clamp to reasonable range to avoid numerical issues
                if (p > 100.0) p = 100.0;
            }
        }
        
        // Parse compute_mode from remaining data
        int64_t compute_mode = 0; // Default: use_mm_for_euclid_dist_if_necessary
        if (offset < Size) {
            compute_mode = Data[offset++] % 3; // Valid values: 0, 1, 2
        }
        
        // Basic cdist with default parameters
        torch::Tensor result1 = torch::cdist(x1, x2);
        
        // cdist with custom p-norm
        torch::Tensor result2 = torch::cdist(x1, x2, p);
        
        // cdist with custom p-norm and compute_mode
        torch::Tensor result3 = torch::cdist(x1, x2, p, compute_mode);
        
        // Test with x1 duplicated (self-distance)
        torch::Tensor result_self = torch::cdist(x1, x1);
        
        // Test different p-norms
        double p_values[] = {0.0, 0.5, 1.0, 2.0, 3.0, std::numeric_limits<double>::infinity()};
        for (double test_p : p_values) {
            try {
                torch::Tensor result_p = torch::cdist(x1, x2, test_p);
            } catch (const std::exception&) {
                // Some p values may not be supported
            }
        }
        
        // Test all compute modes
        for (int64_t mode = 0; mode <= 2; mode++) {
            try {
                torch::Tensor result_mode = torch::cdist(x1, x2, 2.0, mode);
            } catch (const std::exception&) {
                // Some modes may not work for certain tensor configurations
            }
        }
        
        // Try with empty tensors
        if (offset < Size) {
            uint8_t empty_dim = (Data[offset++] % 5) + 1;
            torch::Tensor empty_tensor = torch::empty({0, empty_dim});
            
            try {
                torch::Tensor result_empty = torch::cdist(empty_tensor, x2);
            } catch (const std::exception&) {
                // Expected exception for incompatible shapes
            }
            
            try {
                torch::Tensor result_empty2 = torch::cdist(x1, empty_tensor);
            } catch (const std::exception&) {
                // Expected exception for incompatible shapes
            }
        }
        
        // Try with tensors that have different last dimensions (should throw)
        if (offset + 2 < Size) {
            uint8_t dim1 = (Data[offset++] % 5) + 1;
            uint8_t dim2 = (Data[offset++] % 5) + 1;
            uint8_t dim3 = (Data[offset++] % 5) + 1;
            
            torch::Tensor t1 = torch::rand({dim1, dim3});
            torch::Tensor t2 = torch::rand({dim2, dim3});
            
            // This should work as they have the same last dimension
            torch::Tensor result_diff_batch = torch::cdist(t1, t2);
            
            // Try with different last dimensions (should throw)
            try {
                torch::Tensor t3 = torch::rand({dim1, dim3 + 1});
                torch::Tensor result_incompatible = torch::cdist(t1, t3);
            } catch (const std::exception&) {
                // Expected exception for incompatible shapes
            }
        }
        
        // Test with different dtypes
        try {
            torch::Tensor x1_double = x1.to(torch::kFloat64);
            torch::Tensor x2_double = x2.to(torch::kFloat64);
            torch::Tensor result_double = torch::cdist(x1_double, x2_double);
        } catch (const std::exception&) {
            // May fail for certain configurations
        }
        
        // Test with tensors containing special values
        if (offset < Size && (Data[offset++] % 4) == 0) {
            try {
                torch::Tensor x1_special = x1.clone();
                x1_special.index_put_({0, 0}, std::numeric_limits<float>::infinity());
                torch::Tensor result_special = torch::cdist(x1_special, x2);
            } catch (const std::exception&) {
                // May throw for special values
            }
        }
        
        // Test with higher dimensional tensors (multiple batch dimensions)
        if (offset + 2 < Size) {
            uint8_t b1 = (Data[offset++] % 2) + 1;
            uint8_t b2 = (Data[offset++] % 3) + 1;
            uint8_t b3 = (Data[offset++] % 4) + 1;
            
            try {
                torch::Tensor t1 = torch::rand({b1, b2, feat_dim});
                torch::Tensor t2 = torch::rand({b1, b3, feat_dim});
                torch::Tensor result_batched = torch::cdist(t1, t2);
            } catch (const std::exception&) {
                // May throw for certain shapes
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