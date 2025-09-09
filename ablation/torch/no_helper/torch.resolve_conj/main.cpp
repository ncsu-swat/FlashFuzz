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
        auto tensor_params = generateTensorParams(Data, Size, offset);
        if (!tensor_params.has_value()) {
            return 0;
        }

        auto [shape, dtype, device] = tensor_params.value();
        
        // Skip if shape is empty or too large
        if (shape.empty() || getTotalElements(shape) > MAX_TENSOR_ELEMENTS) {
            return 0;
        }

        // Test with different tensor types, focusing on complex types since resolve_conj
        // is most relevant for complex tensors
        std::vector<torch::ScalarType> test_dtypes = {
            torch::kComplexFloat,
            torch::kComplexDouble,
            torch::kFloat,
            torch::kDouble,
            torch::kInt,
            torch::kLong
        };

        for (auto test_dtype : test_dtypes) {
            // Create input tensor
            torch::Tensor input;
            
            if (test_dtype == torch::kComplexFloat || test_dtype == torch::kComplexDouble) {
                // Create complex tensor with random values
                auto real_part = torch::randn(shape, torch::TensorOptions().dtype(
                    test_dtype == torch::kComplexFloat ? torch::kFloat : torch::kDouble));
                auto imag_part = torch::randn(shape, torch::TensorOptions().dtype(
                    test_dtype == torch::kComplexFloat ? torch::kFloat : torch::kDouble));
                input = torch::complex(real_part, imag_part);
            } else {
                // Create real tensor
                input = torch::randn(shape, torch::TensorOptions().dtype(test_dtype));
            }

            // Test 1: resolve_conj on regular tensor (should return same tensor)
            auto result1 = torch::resolve_conj(input);
            
            // Verify result properties
            if (result1.sizes() != input.sizes()) {
                std::cerr << "Shape mismatch in resolve_conj" << std::endl;
            }
            if (result1.dtype() != input.dtype()) {
                std::cerr << "Dtype mismatch in resolve_conj" << std::endl;
            }
            if (result1.is_conj()) {
                std::cerr << "Result should not have conjugate bit set" << std::endl;
            }

            // Test 2: resolve_conj on conjugated tensor (only for complex types)
            if (test_dtype == torch::kComplexFloat || test_dtype == torch::kComplexDouble) {
                auto conj_input = input.conj();
                
                // Verify conjugate bit is set
                if (!conj_input.is_conj()) {
                    continue; // Skip if conj() didn't set the bit
                }
                
                auto result2 = torch::resolve_conj(conj_input);
                
                // Verify result properties
                if (result2.sizes() != conj_input.sizes()) {
                    std::cerr << "Shape mismatch in resolve_conj with conjugated input" << std::endl;
                }
                if (result2.dtype() != conj_input.dtype()) {
                    std::cerr << "Dtype mismatch in resolve_conj with conjugated input" << std::endl;
                }
                if (result2.is_conj()) {
                    std::cerr << "Result should not have conjugate bit set after resolve_conj" << std::endl;
                }
                
                // Test 3: Double conjugation and resolve
                auto double_conj = conj_input.conj();
                auto result3 = torch::resolve_conj(double_conj);
                
                if (result3.is_conj()) {
                    std::cerr << "Result should not have conjugate bit set after double conj resolve" << std::endl;
                }
            }

            // Test 4: Edge cases with different tensor properties
            
            // Empty tensor
            if (!shape.empty()) {
                auto empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(test_dtype));
                auto empty_result = torch::resolve_conj(empty_tensor);
                if (empty_result.is_conj()) {
                    std::cerr << "Empty tensor result should not have conjugate bit set" << std::endl;
                }
            }

            // Scalar tensor
            auto scalar_tensor = torch::scalar_tensor(1.0, torch::TensorOptions().dtype(test_dtype));
            if (test_dtype == torch::kComplexFloat || test_dtype == torch::kComplexDouble) {
                scalar_tensor = torch::scalar_tensor(std::complex<double>(1.0, 2.0), 
                                                   torch::TensorOptions().dtype(test_dtype));
                auto scalar_conj = scalar_tensor.conj();
                auto scalar_result = torch::resolve_conj(scalar_conj);
                if (scalar_result.is_conj()) {
                    std::cerr << "Scalar result should not have conjugate bit set" << std::endl;
                }
            }
            
            auto scalar_result = torch::resolve_conj(scalar_tensor);
            if (scalar_result.is_conj()) {
                std::cerr << "Scalar result should not have conjugate bit set" << std::endl;
            }

            // Test 5: Chained operations
            if (test_dtype == torch::kComplexFloat || test_dtype == torch::kComplexDouble) {
                auto chained = torch::resolve_conj(torch::resolve_conj(input.conj()));
                if (chained.is_conj()) {
                    std::cerr << "Chained resolve_conj result should not have conjugate bit set" << std::endl;
                }
            }
        }

        // Test 6: Memory layout variations
        if (shape.size() >= 2) {
            auto contiguous_tensor = torch::randn(shape, torch::kComplexFloat);
            auto transposed_tensor = contiguous_tensor.transpose(0, 1);
            
            auto contig_result = torch::resolve_conj(contiguous_tensor.conj());
            auto transposed_result = torch::resolve_conj(transposed_tensor.conj());
            
            if (contig_result.is_conj() || transposed_result.is_conj()) {
                std::cerr << "Memory layout variation results should not have conjugate bit set" << std::endl;
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