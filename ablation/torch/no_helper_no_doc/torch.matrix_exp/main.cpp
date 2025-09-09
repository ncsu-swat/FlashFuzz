#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions and properties
        auto batch_size = extract_int(Data, Size, offset, 1, 4);
        auto matrix_size = extract_int(Data, Size, offset, 2, 8);
        auto dtype_choice = extract_int(Data, Size, offset, 0, 3);
        auto device_choice = extract_int(Data, Size, offset, 0, 1);
        auto requires_grad = extract_bool(Data, Size, offset);

        // Map dtype choice to actual dtype
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat32; break;
        }

        // Map device choice
        torch::Device device = device_choice == 0 ? torch::kCPU : torch::kCUDA;
        
        // Skip CUDA if not available
        if (device.is_cuda() && !torch::cuda::is_available()) {
            device = torch::kCPU;
        }

        // Create input tensor - matrix_exp requires square matrices
        torch::Tensor input;
        
        if (batch_size == 1) {
            // Single matrix case: [matrix_size, matrix_size]
            input = torch::randn({matrix_size, matrix_size}, 
                               torch::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad));
        } else {
            // Batch case: [batch_size, matrix_size, matrix_size]
            input = torch::randn({batch_size, matrix_size, matrix_size}, 
                               torch::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad));
        }

        // Fill tensor with fuzzed data if we have enough bytes
        if (offset < Size) {
            auto flat_input = input.flatten();
            auto num_elements = flat_input.numel();
            
            for (int64_t i = 0; i < num_elements && offset < Size; ++i) {
                float val = extract_float(Data, Size, offset, -10.0f, 10.0f);
                
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                    float imag_val = extract_float(Data, Size, offset, -10.0f, 10.0f);
                    if (dtype == torch::kComplexFloat) {
                        flat_input[i] = std::complex<float>(val, imag_val);
                    } else {
                        flat_input[i] = std::complex<double>(val, imag_val);
                    }
                } else {
                    flat_input[i] = val;
                }
            }
        }

        // Test edge cases with special matrices
        auto edge_case = extract_int(Data, Size, offset, 0, 5);
        switch (edge_case) {
            case 0: {
                // Identity matrix
                input = torch::eye(matrix_size, torch::TensorOptions().dtype(dtype).device(device));
                if (batch_size > 1) {
                    input = input.unsqueeze(0).expand({batch_size, matrix_size, matrix_size});
                }
                break;
            }
            case 1: {
                // Zero matrix
                input = torch::zeros({batch_size > 1 ? batch_size : matrix_size, matrix_size, matrix_size}, 
                                   torch::TensorOptions().dtype(dtype).device(device));
                if (batch_size == 1) {
                    input = input.squeeze(0);
                }
                break;
            }
            case 2: {
                // Diagonal matrix
                auto diag_vals = torch::randn({matrix_size}, torch::TensorOptions().dtype(dtype).device(device));
                input = torch::diag(diag_vals);
                if (batch_size > 1) {
                    input = input.unsqueeze(0).expand({batch_size, matrix_size, matrix_size});
                }
                break;
            }
            case 3: {
                // Symmetric matrix
                auto temp = torch::randn({matrix_size, matrix_size}, torch::TensorOptions().dtype(dtype).device(device));
                input = temp + temp.transpose(0, 1);
                if (batch_size > 1) {
                    input = input.unsqueeze(0).expand({batch_size, matrix_size, matrix_size});
                }
                break;
            }
            case 4: {
                // Skew-symmetric matrix
                auto temp = torch::randn({matrix_size, matrix_size}, torch::TensorOptions().dtype(dtype).device(device));
                input = temp - temp.transpose(0, 1);
                if (batch_size > 1) {
                    input = input.unsqueeze(0).expand({batch_size, matrix_size, matrix_size});
                }
                break;
            }
            default:
                // Keep the random matrix
                break;
        }

        // Set requires_grad after tensor creation if needed
        if (requires_grad && input.is_floating_point()) {
            input.requires_grad_(true);
        }

        // Call matrix_exp
        torch::Tensor result = torch::matrix_exp(input);

        // Verify result properties
        if (result.numel() == 0) {
            return 0;
        }

        // Check that result has same batch dimensions as input
        auto input_sizes = input.sizes();
        auto result_sizes = result.sizes();
        
        if (input_sizes.size() != result_sizes.size()) {
            std::cerr << "Dimension mismatch between input and result" << std::endl;
            return -1;
        }

        for (size_t i = 0; i < input_sizes.size(); ++i) {
            if (input_sizes[i] != result_sizes[i]) {
                std::cerr << "Size mismatch at dimension " << i << std::endl;
                return -1;
            }
        }

        // Test gradient computation if applicable
        if (input.requires_grad() && result.is_floating_point()) {
            auto sum_result = result.sum();
            sum_result.backward();
            
            if (!input.grad().defined()) {
                std::cerr << "Gradient not computed" << std::endl;
                return -1;
            }
        }

        // Additional verification: matrix exponential properties
        // For real matrices, check that result is real
        if (!input.is_complex() && result.is_complex()) {
            std::cerr << "Real input produced complex output unexpectedly" << std::endl;
            return -1;
        }

        // Test with different tensor layouts if we have remaining data
        if (offset < Size) {
            auto layout_test = extract_bool(Data, Size, offset);
            if (layout_test) {
                // Test with contiguous tensor
                auto contiguous_input = input.contiguous();
                torch::Tensor contiguous_result = torch::matrix_exp(contiguous_input);
                
                // Results should be close
                if (!torch::allclose(result, contiguous_result, 1e-5, 1e-5)) {
                    std::cerr << "Contiguous and non-contiguous results differ significantly" << std::endl;
                }
            }
        }

        // Force evaluation to catch any lazy computation issues
        result.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}