#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr, cout
#include <cstring>        // For memcpy

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
        
        // Need enough bytes for tensor creation and parameters
        if (Size < 8)
            return 0;
        
        // Create first input tensor
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have data for second tensor
        if (offset >= Size)
            return 0;
        
        // Create second input tensor
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensors are valid for cosine_similarity
        if (x1.dim() == 0 || x2.dim() == 0)
            return 0;
        
        // Convert to float for cosine_similarity (requires floating point)
        x1 = x1.to(torch::kFloat32);
        x2 = x2.to(torch::kFloat32);
        
        // Get dimension parameter, bounded to valid range
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset] % x1.dim());
            // Handle negative dimensions
            if (offset + 1 < Size && (Data[offset + 1] & 1)) {
                dim = dim - x1.dim(); // Convert to negative indexing
            }
            offset += 2;
        }
        
        // Get eps parameter - use a byte to select from reasonable values
        double eps = 1e-8;
        if (offset < Size) {
            // Map byte to reasonable eps range [1e-12, 1e-4]
            double eps_options[] = {1e-12, 1e-10, 1e-8, 1e-6, 1e-4};
            eps = eps_options[Data[offset] % 5];
            offset++;
        }
        
        // Try to make tensors have compatible shapes along the specified dimension
        // Reshape x2 to match x1's shape if needed
        try {
            // Attempt to broadcast x2 to x1's shape (or vice versa)
            auto x1_sizes = x1.sizes().vec();
            auto x2_sizes = x2.sizes().vec();
            
            // If dimensions don't match, try to expand
            if (x1.dim() != x2.dim()) {
                // Expand the smaller one
                if (x1.dim() < x2.dim()) {
                    while (x1.dim() < x2.dim()) {
                        x1 = x1.unsqueeze(0);
                    }
                } else {
                    while (x2.dim() < x1.dim()) {
                        x2 = x2.unsqueeze(0);
                    }
                }
            }
            
            // Try to expand tensors to be broadcastable
            auto target_sizes = x1.sizes().vec();
            for (int64_t i = 0; i < static_cast<int64_t>(target_sizes.size()); i++) {
                if (i < x2.dim()) {
                    target_sizes[i] = std::max(x1.size(i), x2.size(i));
                }
            }
            
            x1 = x1.expand(target_sizes);
            x2 = x2.expand(target_sizes);
            
            // Ensure dim is valid after potential reshaping
            dim = dim % x1.dim();
            if (dim < 0) dim += x1.dim();
            
        } catch (...) {
            // Shape manipulation failed, continue with original tensors
            // Let cosine_similarity handle any remaining shape issues
        }
        
        // Apply cosine_similarity
        try {
            torch::Tensor result = torch::cosine_similarity(x1, x2, dim, eps);
            
            // Access result to ensure computation happens
            volatile float check = result.sum().item<float>();
            (void)check;
        } catch (const c10::Error& e) {
            // Expected failures (shape mismatches, etc.) - silently catch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // Keep the input
}