#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor (x)
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // nextafter requires floating point tensors
        if (!x.is_floating_point()) {
            // Convert to float for testing
            x = x.to(torch::kFloat);
        }
        
        // Create second tensor (other)
        torch::Tensor other;
        if (offset < Size) {
            other = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data for a second tensor, use the same tensor
            other = x.clone();
        }
        
        // Ensure other is also floating point
        if (!other.is_floating_point()) {
            other = other.to(torch::kFloat);
        }
        
        // Try different variants of nextafter
        
        // 1. Basic nextafter
        torch::Tensor result1 = torch::nextafter(x, other);
        
        // 2. Out variant
        try {
            torch::Tensor out = torch::empty_like(x);
            torch::nextafter_out(out, x, other);
        } catch (...) {
            // Shape mismatch or other expected failures
        }
        
        // 3. In-place variant (nextafter_)
        try {
            torch::Tensor x_copy = x.clone();
            x_copy.nextafter_(other);
        } catch (...) {
            // May fail due to shape mismatch
        }
        
        // 4. Try with scalar tensors
        if (x.numel() > 0) {
            try {
                double scalar_value = x.flatten()[0].item<double>();
                torch::Tensor scalar_tensor = torch::tensor(scalar_value);
                torch::Tensor result_scalar = torch::nextafter(x, scalar_tensor);
            } catch (...) {
                // Expected failure for some cases
            }
            
            // Test nextafter with scalar as first argument
            try {
                if (other.numel() > 0) {
                    double other_scalar = other.flatten()[0].item<double>();
                    torch::Tensor scalar_x = torch::tensor(1.0);
                    torch::Tensor scalar_other = torch::tensor(other_scalar);
                    torch::Tensor result_scalar_first = torch::nextafter(scalar_x, scalar_other);
                }
            } catch (...) {
                // Expected failure
            }
        }
        
        // 5. Try with different floating point dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            torch::Dtype target_dtype;
            
            switch (dtype_selector) {
                case 0: target_dtype = torch::kFloat; break;
                case 1: target_dtype = torch::kDouble; break;
                case 2: target_dtype = torch::kHalf; break;
                case 3: target_dtype = torch::kBFloat16; break;
                default: target_dtype = torch::kFloat; break;
            }
            
            try {
                torch::Tensor x_cast = x.to(target_dtype);
                torch::Tensor other_cast = other.to(target_dtype);
                torch::Tensor result_cast = torch::nextafter(x_cast, other_cast);
            } catch (...) {
                // Some dtype combinations may not be supported
            }
        }
        
        // 6. Try with empty tensors
        try {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat);
            torch::Tensor result_empty = torch::nextafter(empty_tensor, empty_tensor);
        } catch (...) {
            // Expected to fail in some cases
        }
        
        // 7. Test broadcasting behavior with different shapes
        if (offset + 2 < Size) {
            try {
                // Create a tensor that can broadcast with x
                std::vector<int64_t> broadcast_shape;
                if (x.dim() > 0) {
                    // Use last dimension for broadcasting
                    broadcast_shape.push_back(x.size(-1));
                } else {
                    broadcast_shape.push_back(1);
                }
                torch::Tensor broadcast_other = torch::randn(broadcast_shape);
                torch::Tensor result_broadcast = torch::nextafter(x, broadcast_other);
            } catch (...) {
                // Broadcasting may fail
            }
        }
        
        // 8. Test with special floating point values
        try {
            torch::Tensor inf_tensor = torch::tensor({std::numeric_limits<float>::infinity()});
            torch::Tensor neg_inf_tensor = torch::tensor({-std::numeric_limits<float>::infinity()});
            torch::Tensor nan_tensor = torch::tensor({std::numeric_limits<float>::quiet_NaN()});
            torch::Tensor zero_tensor = torch::tensor({0.0f});
            
            torch::nextafter(x.flatten().slice(0, 0, 1), inf_tensor);
            torch::nextafter(x.flatten().slice(0, 0, 1), neg_inf_tensor);
            torch::nextafter(x.flatten().slice(0, 0, 1), nan_tensor);
            torch::nextafter(x.flatten().slice(0, 0, 1), zero_tensor);
            torch::nextafter(zero_tensor, x.flatten().slice(0, 0, 1));
        } catch (...) {
            // Some special value operations may fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}