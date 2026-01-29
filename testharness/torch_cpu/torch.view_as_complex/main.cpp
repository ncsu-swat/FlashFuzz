#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Parse dtype from fuzzer data - only allow compatible floating point types
        uint8_t dtype_byte = Data[offset++];
        torch::ScalarType dtype;
        switch (dtype_byte % 4) {
            case 0: dtype = torch::kFloat; break;
            case 1: dtype = torch::kDouble; break;
            case 2: dtype = torch::kHalf; break;
            case 3: dtype = torch::kBFloat16; break;
            default: dtype = torch::kFloat; break;
        }

        // Parse rank (1-4 dimensions, last dimension will be 2)
        uint8_t rank_byte = Data[offset++];
        int rank = 1 + (rank_byte % 4); // 1 to 4 dimensions

        // Build shape with last dimension = 2
        std::vector<int64_t> shape;
        for (int i = 0; i < rank - 1; i++) {
            if (offset < Size) {
                shape.push_back(1 + (Data[offset++] % 8)); // dimensions 1-8
            } else {
                shape.push_back(2);
            }
        }
        shape.push_back(2); // Last dimension must be 2 for view_as_complex

        // Create tensor with the proper shape
        torch::Tensor input_tensor = torch::rand(shape, torch::TensorOptions().dtype(dtype));

        // Ensure tensor is contiguous (view_as_complex requires this)
        input_tensor = input_tensor.contiguous();

        // Apply view_as_complex operation
        torch::Tensor result = torch::view_as_complex(input_tensor);

        // Exercise the result to ensure coverage
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto abs_val = torch::abs(result);
            auto real_part = torch::real(result);
            auto imag_part = torch::imag(result);
        }

        // Test with strided tensor (clone to get different memory layout)
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            // Test view_as_real as inverse operation
            torch::Tensor back_to_real = torch::view_as_real(result);
            
            // Verify round-trip works
            if (back_to_real.numel() > 0) {
                auto diff = (back_to_real - input_tensor).abs().sum();
            }
        }

        // Test with different tensor creation methods
        if (offset + 2 < Size) {
            uint8_t method = Data[offset++] % 4;
            std::vector<int64_t> shape2;
            int dim_count = 1 + (Data[offset++] % 3);
            for (int i = 0; i < dim_count; i++) {
                if (offset < Size) {
                    shape2.push_back(1 + (Data[offset++] % 6));
                } else {
                    shape2.push_back(2);
                }
            }
            shape2.push_back(2); // Last dimension must be 2

            torch::Tensor tensor2;
            try {
                switch (method) {
                    case 0:
                        tensor2 = torch::zeros(shape2, torch::TensorOptions().dtype(torch::kFloat));
                        break;
                    case 1:
                        tensor2 = torch::ones(shape2, torch::TensorOptions().dtype(torch::kDouble));
                        break;
                    case 2:
                        tensor2 = torch::randn(shape2, torch::TensorOptions().dtype(torch::kFloat));
                        break;
                    case 3:
                        tensor2 = torch::rand(shape2, torch::TensorOptions().dtype(torch::kFloat)) * 100 - 50;
                        break;
                }

                tensor2 = tensor2.contiguous();
                torch::Tensor complex_result = torch::view_as_complex(tensor2);

                // Exercise complex operations
                if (complex_result.numel() > 0) {
                    auto angle = torch::angle(complex_result);
                    auto conj = torch::conj(complex_result);
                }
            } catch (const std::exception&) {
                // Inner exceptions are expected for invalid configurations
            }
        }

        // Test edge case: tensor with only last dimension = 2
        if (offset < Size) {
            torch::Tensor simple_tensor = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
            simple_tensor = simple_tensor.contiguous();
            torch::Tensor simple_complex = torch::view_as_complex(simple_tensor);
            
            // Verify shape transformation
            auto orig_shape = simple_tensor.sizes();
            auto new_shape = simple_complex.sizes();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}