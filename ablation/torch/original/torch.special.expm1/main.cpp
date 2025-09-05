#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <limits>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: dtype, rank, and operation mode
        if (Size < 3) {
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get operation mode if we have extra bytes
        uint8_t op_mode = 0;
        if (offset < Size) {
            op_mode = Data[offset++];
        }
        
        // Test different scenarios based on op_mode
        switch (op_mode % 8) {
            case 0: {
                // Basic expm1 operation
                torch::Tensor result = torch::special::expm1(input);
                
                // Verify shape preservation
                if (result.sizes() != input.sizes()) {
                    std::cerr << "Shape mismatch: input " << input.sizes() 
                             << " vs result " << result.sizes() << std::endl;
                }
                break;
            }
            
            case 1: {
                // Test with pre-allocated output tensor
                torch::Tensor out = torch::empty_like(input);
                torch::special::expm1_out(out, input);
                
                // Verify in-place operation worked
                torch::Tensor result = torch::special::expm1(input);
                if (!torch::allclose(out, result, 1e-5, 1e-8)) {
                    std::cerr << "Output tensor mismatch" << std::endl;
                }
                break;
            }
            
            case 2: {
                // Test with different memory layouts (non-contiguous)
                if (input.numel() > 1 && input.dim() > 0) {
                    torch::Tensor transposed = input.dim() >= 2 ? 
                        input.transpose(0, input.dim() - 1) : input;
                    torch::Tensor result = torch::special::expm1(transposed);
                    
                    // Verify operation works on non-contiguous tensors
                    if (result.is_contiguous() == transposed.is_contiguous()) {
                        // Expected behavior - output contiguity may differ
                    }
                }
                break;
            }
            
            case 3: {
                // Test with special values
                if (input.numel() > 0 && input.dtype() == torch::kFloat || 
                    input.dtype() == torch::kDouble) {
                    // Create tensor with special values
                    torch::Tensor special_vals = torch::zeros_like(input);
                    if (offset < Size) {
                        uint8_t special_type = Data[offset++] % 5;
                        switch (special_type) {
                            case 0: special_vals.fill_(0.0); break;  // expm1(0) = 0
                            case 1: special_vals.fill_(-1.0); break; // expm1(-1) = -0.632...
                            case 2: special_vals.fill_(std::numeric_limits<float>::infinity()); break;
                            case 3: special_vals.fill_(-std::numeric_limits<float>::infinity()); break;
                            case 4: special_vals.fill_(std::numeric_limits<float>::quiet_NaN()); break;
                        }
                        torch::Tensor result = torch::special::expm1(special_vals);
                    }
                }
                break;
            }
            
            case 4: {
                // Test precision for small values (main advantage of expm1)
                if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) {
                    // Scale input to small values where expm1 has precision advantage
                    torch::Tensor small_input = input * 1e-10;
                    torch::Tensor expm1_result = torch::special::expm1(small_input);
                    torch::Tensor exp_minus_1 = torch::exp(small_input) - 1;
                    
                    // expm1 should be more accurate for small values
                    // Just compute both to exercise the paths
                }
                break;
            }
            
            case 5: {
                // Test with complex numbers if applicable
                if (input.dtype() == torch::kComplexFloat || 
                    input.dtype() == torch::kComplexDouble) {
                    torch::Tensor result = torch::special::expm1(input);
                    
                    // Complex expm1: e^(a+bi) - 1 = e^a * (cos(b) + i*sin(b)) - 1
                    // Just verify it doesn't crash
                }
                break;
            }
            
            case 6: {
                // Test with views and slices
                if (input.numel() > 2) {
                    // Create a view
                    torch::Tensor view = input.view(-1);
                    torch::Tensor result = torch::special::expm1(view);
                    
                    // Test with slice
                    if (input.dim() > 0 && input.size(0) > 1) {
                        torch::Tensor slice = input.narrow(0, 0, 1);
                        torch::Tensor slice_result = torch::special::expm1(slice);
                    }
                }
                break;
            }
            
            case 7: {
                // Test gradient computation if floating point
                if ((input.dtype() == torch::kFloat || input.dtype() == torch::kDouble) 
                    && input.numel() > 0) {
                    // Set requires_grad
                    torch::Tensor grad_input = input.detach().requires_grad_(true);
                    torch::Tensor result = torch::special::expm1(grad_input);
                    
                    // Compute gradient: d/dx(e^x - 1) = e^x
                    if (result.numel() > 0) {
                        torch::Tensor grad_out = torch::ones_like(result);
                        result.backward(grad_out);
                        
                        // Gradient should be exp(input)
                        torch::Tensor expected_grad = torch::exp(grad_input);
                        // Just compute to exercise the path
                    }
                }
                break;
            }
        }
        
        // Additional edge case testing with remaining bytes
        while (offset + 1 < Size) {
            uint8_t extra_test = Data[offset++];
            
            switch (extra_test % 4) {
                case 0: {
                    // Test empty tensor
                    torch::Tensor empty = torch::empty({0});
                    torch::Tensor empty_result = torch::special::expm1(empty);
                    break;
                }
                case 1: {
                    // Test scalar tensor
                    torch::Tensor scalar = torch::tensor(3.14159);
                    torch::Tensor scalar_result = torch::special::expm1(scalar);
                    break;
                }
                case 2: {
                    // Test large tensor if we have enough memory
                    if (offset < Size) {
                        uint8_t size_selector = Data[offset++] % 4;
                        std::vector<int64_t> large_shape;
                        switch (size_selector) {
                            case 0: large_shape = {1000}; break;
                            case 1: large_shape = {100, 10}; break;
                            case 2: large_shape = {10, 10, 10}; break;
                            case 3: large_shape = {5, 5, 5, 5}; break;
                        }
                        torch::Tensor large = torch::randn(large_shape);
                        torch::Tensor large_result = torch::special::expm1(large);
                    }
                    break;
                }
                case 3: {
                    // Test with different devices if available
                    #ifdef USE_GPU
                    if (torch::cuda::is_available() && input.numel() > 0) {
                        torch::Tensor cuda_input = input.to(torch::kCUDA);
                        torch::Tensor cuda_result = torch::special::expm1(cuda_input);
                        torch::Tensor cpu_result = cuda_result.to(torch::kCPU);
                    }
                    #endif
                    break;
                }
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        // Catch any other unexpected exceptions
        std::cout << "Unknown exception caught" << std::endl;
        return -1;
    }
    
    return 0;
}