#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data for tensor creation and parameters
        if (Size < 16) {
            return 0;
        }

        // Extract parameters for tensor creation
        auto shape_info = extract_tensor_shape(Data, Size, offset);
        auto dtype_info = extract_dtype(Data, Size, offset);
        
        // Create real and imaginary tensors with various dtypes
        torch::Tensor real_tensor, imag_tensor;
        
        // Test with different floating point dtypes that support complex operations
        std::vector<torch::ScalarType> valid_dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kFloat16
        };
        
        auto selected_dtype = valid_dtypes[extract_int(Data, Size, offset) % valid_dtypes.size()];
        
        // Create tensors with the selected dtype
        real_tensor = create_tensor(shape_info, selected_dtype);
        imag_tensor = create_tensor(shape_info, selected_dtype);
        
        // Fill tensors with fuzzed data
        fill_tensor_with_data(real_tensor, Data, Size, offset);
        fill_tensor_with_data(imag_tensor, Data, Size, offset);
        
        // Test torch::complex with two tensor arguments
        auto complex_result1 = torch::complex(real_tensor, imag_tensor);
        
        // Verify the result has complex dtype
        if (!complex_result1.is_complex()) {
            std::cerr << "Expected complex tensor but got non-complex" << std::endl;
        }
        
        // Test with scalar values
        if (offset + 16 < Size) {
            double real_scalar = extract_double(Data, Size, offset);
            double imag_scalar = extract_double(Data, Size, offset);
            
            // Test torch::complex with scalar arguments
            auto complex_scalar = torch::complex(torch::tensor(real_scalar), torch::tensor(imag_scalar));
            
            // Test mixed tensor and scalar
            auto complex_mixed1 = torch::complex(real_tensor, torch::tensor(imag_scalar));
            auto complex_mixed2 = torch::complex(torch::tensor(real_scalar), imag_tensor);
        }
        
        // Test edge cases with special values
        if (offset + 4 < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 8) {
                case 0: {
                    // Test with zeros
                    auto zero_real = torch::zeros_like(real_tensor);
                    auto zero_imag = torch::zeros_like(imag_tensor);
                    auto complex_zeros = torch::complex(zero_real, zero_imag);
                    break;
                }
                case 1: {
                    // Test with ones
                    auto ones_real = torch::ones_like(real_tensor);
                    auto ones_imag = torch::ones_like(imag_tensor);
                    auto complex_ones = torch::complex(ones_real, ones_imag);
                    break;
                }
                case 2: {
                    // Test with infinity
                    auto inf_real = torch::full_like(real_tensor, std::numeric_limits<float>::infinity());
                    auto inf_imag = torch::full_like(imag_tensor, std::numeric_limits<float>::infinity());
                    auto complex_inf = torch::complex(inf_real, inf_imag);
                    break;
                }
                case 3: {
                    // Test with negative infinity
                    auto ninf_real = torch::full_like(real_tensor, -std::numeric_limits<float>::infinity());
                    auto ninf_imag = torch::full_like(imag_tensor, -std::numeric_limits<float>::infinity());
                    auto complex_ninf = torch::complex(ninf_real, ninf_imag);
                    break;
                }
                case 4: {
                    // Test with NaN
                    auto nan_real = torch::full_like(real_tensor, std::numeric_limits<float>::quiet_NaN());
                    auto nan_imag = torch::full_like(imag_tensor, std::numeric_limits<float>::quiet_NaN());
                    auto complex_nan = torch::complex(nan_real, nan_imag);
                    break;
                }
                case 5: {
                    // Test with very large values
                    auto large_real = torch::full_like(real_tensor, 1e30f);
                    auto large_imag = torch::full_like(imag_tensor, 1e30f);
                    auto complex_large = torch::complex(large_real, large_imag);
                    break;
                }
                case 6: {
                    // Test with very small values
                    auto small_real = torch::full_like(real_tensor, 1e-30f);
                    auto small_imag = torch::full_like(imag_tensor, 1e-30f);
                    auto complex_small = torch::complex(small_real, small_imag);
                    break;
                }
                case 7: {
                    // Test with mismatched shapes (should handle broadcasting)
                    if (real_tensor.numel() > 1) {
                        auto scalar_imag = torch::tensor(1.0f);
                        auto complex_broadcast = torch::complex(real_tensor, scalar_imag);
                    }
                    break;
                }
            }
        }
        
        // Test with different tensor properties
        if (offset + 2 < Size) {
            bool requires_grad = (Data[offset++] % 2) == 1;
            bool pin_memory = (Data[offset++] % 2) == 1;
            
            if (requires_grad) {
                real_tensor.requires_grad_(true);
                imag_tensor.requires_grad_(true);
                auto complex_grad = torch::complex(real_tensor, imag_tensor);
            }
        }
        
        // Test operations on the resulting complex tensor
        if (complex_result1.numel() > 0) {
            // Test basic operations that should work with complex tensors
            auto real_part = torch::real(complex_result1);
            auto imag_part = torch::imag(complex_result1);
            auto abs_result = torch::abs(complex_result1);
            auto conj_result = torch::conj(complex_result1);
            
            // Test arithmetic operations
            if (offset + 1 < Size) {
                uint8_t op_type = Data[offset++] % 4;
                switch (op_type) {
                    case 0:
                        if (complex_result1.numel() > 1) {
                            auto add_result = complex_result1 + complex_result1;
                        }
                        break;
                    case 1:
                        if (complex_result1.numel() > 1) {
                            auto sub_result = complex_result1 - complex_result1;
                        }
                        break;
                    case 2:
                        if (complex_result1.numel() > 1) {
                            auto mul_result = complex_result1 * complex_result1;
                        }
                        break;
                    case 3:
                        if (complex_result1.numel() > 1) {
                            auto div_result = complex_result1 / (complex_result1 + torch::tensor(std::complex<float>(1.0f, 0.0f)));
                        }
                        break;
                }
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