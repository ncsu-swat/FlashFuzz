#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Need at least some data to create a tensor
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for pdist
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // pdist requires a 2D floating point tensor with shape [N, M] where N >= 1 and M >= 1
        // Convert to float if not already floating point
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // If the tensor doesn't have the right shape, reshape it
        if (input.dim() != 2) {
            if (input.numel() == 0) {
                input = torch::ones({2, 2}, torch::kFloat32);
            } else {
                int64_t numel = input.numel();
                int64_t dim1 = std::max(static_cast<int64_t>(2), static_cast<int64_t>(std::sqrt(numel)));
                int64_t dim2 = std::max(static_cast<int64_t>(1), numel / dim1);
                // Flatten and take what we need
                input = input.flatten().slice(0, 0, dim1 * dim2).reshape({dim1, dim2});
            }
        }
        
        // Ensure at least 2 rows for meaningful pdist output
        if (input.size(0) < 2) {
            input = torch::cat({input, input}, 0);
        }
        
        // Get a p-norm value from the input data
        double p = 2.0; // Default to Euclidean distance
        if (offset < Size) {
            uint8_t p_byte = Data[offset++];
            
            // Map to common p-norm values
            switch (p_byte % 6) {
                case 0: p = 0.5; break;  // Fractional p
                case 1: p = 1.0; break;  // Manhattan distance
                case 2: p = 2.0; break;  // Euclidean distance
                case 3: p = 3.0; break;  // Higher order norm
                case 4: p = std::numeric_limits<double>::infinity(); break; // Infinity norm
                case 5: 
                    // Use a positive value from the data
                    if (offset < Size) {
                        p = 0.1 + static_cast<double>(Data[offset++]) / 25.5; // Range [0.1, 10.1]
                    }
                    break;
            }
        }
        
        // Apply pdist operation
        torch::Tensor output = torch::pdist(input, p);
        
        // Try with different p values if we have more data
        if (offset + 1 < Size) {
            try {
                double p2 = 0.1 + static_cast<double>(Data[offset++]) / 25.5;
                torch::Tensor output2 = torch::pdist(input, p2);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test edge cases with specific shapes if we have enough data
        if (offset + 2 < Size) {
            uint8_t shape_selector = Data[offset++];
            
            torch::Tensor edge_input;
            
            switch (shape_selector % 5) {
                case 0:
                    // Two rows (minimal for non-empty output)
                    edge_input = torch::randn({2, 3}, torch::kFloat32);
                    break;
                case 1:
                    // Two identical rows (should result in zero distances)
                    edge_input = torch::ones({2, 3}, torch::kFloat32);
                    break;
                case 2:
                    // Moderate number of rows
                    edge_input = torch::randn({10, 4}, torch::kFloat32);
                    break;
                case 3:
                    // Higher dimensional features
                    edge_input = torch::randn({5, 20}, torch::kFloat32);
                    break;
                case 4:
                    // Minimal valid case
                    edge_input = torch::randn({2, 1}, torch::kFloat32);
                    break;
            }
            
            try {
                torch::Tensor edge_output = torch::pdist(edge_input, p);
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with contiguous vs non-contiguous tensors
        if (offset < Size && (Data[offset] % 2 == 0)) {
            try {
                torch::Tensor transposed = input.t().contiguous().t();
                torch::Tensor output_t = torch::pdist(transposed, p);
            } catch (...) {
                // Silently ignore
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