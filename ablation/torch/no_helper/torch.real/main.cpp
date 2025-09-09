#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate tensor parameters
        auto shape = generateRandomShape(Data, Size, offset, 1, 6);
        if (shape.empty()) return 0;

        // Generate dtype - focus on complex types since real() is most meaningful for them
        std::vector<torch::ScalarType> complex_dtypes = {
            torch::kComplexFloat, torch::kComplexDouble, torch::kComplexHalf
        };
        std::vector<torch::ScalarType> real_dtypes = {
            torch::kFloat, torch::kDouble, torch::kHalf, torch::kInt, torch::kLong
        };
        
        // Mix complex and real types to test edge cases
        std::vector<torch::ScalarType> all_dtypes = complex_dtypes;
        all_dtypes.insert(all_dtypes.end(), real_dtypes.begin(), real_dtypes.end());
        
        auto dtype = generateRandomChoice(Data, Size, offset, all_dtypes);

        // Generate device
        auto device = generateRandomDevice(Data, Size, offset);

        // Create input tensor with various initialization strategies
        torch::Tensor input;
        auto init_strategy = generateRandomInt(Data, Size, offset, 0, 4);
        
        switch (init_strategy) {
            case 0:
                // Random values
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble || dtype == torch::kComplexHalf) {
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                } else {
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                }
                break;
            case 1:
                // Zeros
                input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).device(device));
                break;
            case 2:
                // Ones
                input = torch::ones(shape, torch::TensorOptions().dtype(dtype).device(device));
                break;
            case 3:
                // Complex tensor with specific real/imaginary parts
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble || dtype == torch::kComplexHalf) {
                    auto real_part = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat).device(device));
                    auto imag_part = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat).device(device));
                    input = torch::complex(real_part, imag_part).to(dtype);
                } else {
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                }
                break;
            default:
                // Edge values (inf, nan for floating types)
                input = torch::randn(shape, torch::TensorOptions().dtype(dtype).device(device));
                if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                    // Add some inf/nan values
                    auto mask = torch::rand(shape, torch::TensorOptions().device(device)) < 0.1;
                    input = torch::where(mask, torch::complex(torch::tensor(std::numeric_limits<float>::infinity()), 
                                                            torch::tensor(std::numeric_limits<float>::quiet_NaN())), input);
                }
                break;
        }

        // Test torch::real function
        torch::Tensor result = torch::real(input);

        // Verify basic properties
        if (result.numel() != input.numel()) {
            std::cerr << "Error: Result tensor has different number of elements" << std::endl;
        }

        if (result.sizes() != input.sizes()) {
            std::cerr << "Error: Result tensor has different shape" << std::endl;
        }

        // For complex input, verify the result is the real part
        if (input.is_complex()) {
            // The result should be real-valued
            if (result.is_complex()) {
                std::cerr << "Error: Real of complex tensor should not be complex" << std::endl;
            }
            
            // Check that result dtype is the corresponding real type
            auto expected_real_dtype = torch::kFloat;
            if (input.scalar_type() == torch::kComplexDouble) {
                expected_real_dtype = torch::kDouble;
            } else if (input.scalar_type() == torch::kComplexHalf) {
                expected_real_dtype = torch::kHalf;
            }
            
            if (result.scalar_type() != expected_real_dtype) {
                std::cerr << "Error: Unexpected result dtype for complex input" << std::endl;
            }
        } else {
            // For real input, the result should be the same as input
            if (!torch::allclose(result, input, 1e-6, 1e-6, /*equal_nan=*/true)) {
                std::cerr << "Warning: Real of real tensor differs from input" << std::endl;
            }
        }

        // Test edge cases with different tensor properties
        if (generateRandomBool(Data, Size, offset)) {
            // Test with requires_grad
            if (input.is_floating_point() || input.is_complex()) {
                input.requires_grad_(true);
                auto grad_result = torch::real(input);
                
                // Test backward pass if result requires grad
                if (grad_result.requires_grad() && grad_result.numel() > 0) {
                    auto loss = grad_result.sum();
                    loss.backward();
                }
            }
        }

        // Test with different memory layouts
        if (generateRandomBool(Data, Size, offset) && input.dim() >= 2) {
            auto transposed = input.transpose(0, 1);
            auto transposed_result = torch::real(transposed);
            
            // Verify shape consistency
            if (transposed_result.sizes() != transposed.sizes()) {
                std::cerr << "Error: Shape mismatch with transposed input" << std::endl;
            }
        }

        // Test with sliced tensors
        if (generateRandomBool(Data, Size, offset) && input.numel() > 1) {
            auto sliced = input.slice(0, 0, std::min(2L, input.size(0)));
            auto sliced_result = torch::real(sliced);
            
            if (sliced_result.sizes() != sliced.sizes()) {
                std::cerr << "Error: Shape mismatch with sliced input" << std::endl;
            }
        }

        // Test storage sharing property (mentioned in documentation)
        if (input.is_complex()) {
            // For complex tensors, real() should share storage with input
            // This is implementation-specific and may not always be testable
            auto result_storage_size = result.storage().nbytes();
            auto input_storage_size = input.storage().nbytes();
            
            // Basic sanity check - result storage should be reasonable
            if (result_storage_size == 0 && result.numel() > 0) {
                std::cerr << "Warning: Result has zero storage size but non-zero elements" << std::endl;
            }
        }

        // Test with empty tensor
        if (generateRandomBool(Data, Size, offset)) {
            auto empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype).device(device));
            auto empty_result = torch::real(empty_tensor);
            
            if (empty_result.numel() != 0) {
                std::cerr << "Error: Real of empty tensor should be empty" << std::endl;
            }
        }

        // Test with scalar tensor
        if (generateRandomBool(Data, Size, offset)) {
            torch::Tensor scalar_tensor;
            if (dtype == torch::kComplexFloat) {
                scalar_tensor = torch::tensor(std::complex<float>(1.5f, 2.5f), torch::TensorOptions().device(device));
            } else if (dtype == torch::kComplexDouble) {
                scalar_tensor = torch::tensor(std::complex<double>(1.5, 2.5), torch::TensorOptions().device(device));
            } else {
                scalar_tensor = torch::tensor(1.5, torch::TensorOptions().dtype(dtype).device(device));
            }
            
            auto scalar_result = torch::real(scalar_tensor);
            
            if (scalar_result.dim() != 0) {
                std::cerr << "Error: Real of scalar should be scalar" << std::endl;
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}