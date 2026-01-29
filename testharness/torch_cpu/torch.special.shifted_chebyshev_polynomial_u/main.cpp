#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor x (values should ideally be in valid range for the polynomial)
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float for computation
        if (!x.is_floating_point()) {
            x = x.to(torch::kFloat32);
        }
        
        // Parse n value (degree of the polynomial) from remaining data
        int64_t n_val = 0;
        if (offset < Size) {
            n_val = Data[offset++] % 20;  // Limit to 0-19 for reasonable computation
        }
        
        // Create n as a scalar tensor for the API
        torch::Tensor n = torch::tensor(n_val, torch::kInt64);
        
        // Apply the shifted_chebyshev_polynomial_u operation with tensor inputs
        try {
            torch::Tensor result = torch::special::shifted_chebyshev_polynomial_u(x, n);
            
            // Force computation
            if (result.numel() > 0) {
                volatile float val = result.sum().item<float>();
                (void)val;
            }
        } catch (const c10::Error&) {
            // Expected for invalid inputs - silently continue
        }
        
        // Try with a tensor of n values (broadcasting case)
        if (offset + 2 < Size) {
            try {
                // Create a small tensor for n with integer type
                uint8_t n_size = (Data[offset++] % 4) + 1;  // 1-4 elements
                std::vector<int64_t> n_values;
                for (int i = 0; i < n_size && offset < Size; i++) {
                    n_values.push_back(Data[offset++] % 15);  // n values 0-14
                }
                torch::Tensor n_tensor = torch::tensor(n_values, torch::kInt64);
                
                // Create compatible x tensor
                torch::Tensor x2 = torch::rand({static_cast<int64_t>(n_values.size())});
                
                torch::Tensor result2 = torch::special::shifted_chebyshev_polynomial_u(x2, n_tensor);
                
                // Force computation
                if (result2.numel() > 0) {
                    volatile float val = result2.sum().item<float>();
                    (void)val;
                }
            } catch (const c10::Error&) {
                // Expected for shape mismatches - silently continue
            }
        }
        
        // Try with different x tensor shapes
        if (offset + 2 < Size) {
            try {
                torch::Tensor x3 = fuzzer_utils::createTensor(Data, Size, offset);
                if (!x3.is_floating_point()) {
                    x3 = x3.to(torch::kFloat32);
                }
                
                int64_t n_val2 = (offset < Size) ? (Data[offset++] % 10) : 0;
                torch::Tensor n2 = torch::tensor(n_val2, torch::kInt64);
                
                torch::Tensor result3 = torch::special::shifted_chebyshev_polynomial_u(x3, n2);
                
                // Force computation
                if (result3.numel() > 0) {
                    volatile float val = result3.sum().item<float>();
                    (void)val;
                }
            } catch (const c10::Error&) {
                // Expected for invalid inputs - silently continue
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}