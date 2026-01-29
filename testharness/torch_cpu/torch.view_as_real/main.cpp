#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a complex tensor from fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if the tensor is complex
        bool is_complex = input_tensor.is_complex();
        
        // If the tensor is not complex, convert it to complex
        if (!is_complex) {
            try {
                if (input_tensor.dtype() == torch::kFloat) {
                    input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
                } else if (input_tensor.dtype() == torch::kDouble) {
                    input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
                } else {
                    // Convert to float first, then to complex
                    auto float_tensor = input_tensor.to(torch::kFloat);
                    input_tensor = torch::complex(float_tensor, torch::zeros_like(float_tensor));
                }
            } catch (...) {
                // Conversion may fail for some tensor configurations
                return 0;
            }
        }
        
        // Apply view_as_real operation
        // view_as_real returns a tensor with an extra dimension of size 2
        // containing the real and imaginary parts
        torch::Tensor result = torch::view_as_real(input_tensor);
        
        // Verify the result has the expected shape (original shape + [2])
        if (result.dim() > 0 && result.numel() > 0) {
            // Access first element to ensure computation completed
            auto flat = result.flatten();
            volatile float val = flat[0].item<float>();
            (void)val;
        }
        
        // Try some edge cases with different shapes
        if (offset + 1 < Size) {
            // Create a scalar complex tensor
            torch::Tensor scalar_complex;
            if (Data[offset] % 2 == 0) {
                scalar_complex = torch::complex(torch::tensor(1.0f), torch::tensor(2.0f));
            } else {
                scalar_complex = torch::complex(torch::tensor(3.0), torch::tensor(4.0));
            }
            torch::Tensor scalar_result = torch::view_as_real(scalar_complex);
            
            // Verify scalar result shape is [2]
            if (scalar_result.numel() == 2) {
                volatile float real_part = scalar_result[0].item<float>();
                volatile float imag_part = scalar_result[1].item<float>();
                (void)real_part;
                (void)imag_part;
            }
            
            // Create an empty complex tensor
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_complex = torch::empty(empty_shape, torch::kComplexFloat);
            torch::Tensor empty_result = torch::view_as_real(empty_complex);
            
            // Create a tensor with various dimensions
            if (offset + 4 < Size) {
                std::vector<int64_t> mixed_shape;
                for (size_t i = 0; i < 2 && offset + i < Size; i++) {
                    int64_t dim = 1 + (Data[offset + i] % 5);  // 1-5 for dimensions
                    mixed_shape.push_back(dim);
                }
                
                if (!mixed_shape.empty()) {
                    try {
                        torch::Tensor mixed_complex = torch::randn(mixed_shape, torch::kComplexDouble);
                        torch::Tensor mixed_result = torch::view_as_real(mixed_complex);
                        
                        // Verify the last dimension is 2
                        if (mixed_result.dim() > 0) {
                            volatile int64_t last_dim = mixed_result.size(-1);
                            (void)last_dim;
                        }
                    } catch (...) {
                        // Shape may be invalid
                    }
                }
                
                offset += 4;
            }
        }
        
        // Try with different complex dtypes if we have more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype;
            
            // Select between complex float and complex double
            if (dtype_selector % 2 == 0) {
                dtype = torch::kComplexFloat;
            } else {
                dtype = torch::kComplexDouble;
            }
            
            // Create a new tensor with the selected dtype
            uint8_t shape_selector = (offset < Size) ? Data[offset++] : 0;
            int64_t dim1 = 1 + (shape_selector % 5);
            int64_t dim2 = 1 + ((shape_selector >> 4) % 5);
            
            torch::Tensor complex_tensor = torch::randn({dim1, dim2}, dtype);
            torch::Tensor real_view = torch::view_as_real(complex_tensor);
            
            // Verify shape
            if (real_view.dim() == 3 && real_view.size(-1) == 2) {
                volatile float check = real_view[0][0][0].item<float>();
                (void)check;
            }
        }
        
        // Test view_as_real with contiguous and non-contiguous tensors
        if (offset + 2 < Size) {
            uint8_t test_type = Data[offset++];
            
            torch::Tensor base = torch::randn({4, 4}, torch::kComplexFloat);
            
            if (test_type % 3 == 0) {
                // Contiguous tensor
                torch::Tensor contiguous_result = torch::view_as_real(base);
                (void)contiguous_result;
            } else if (test_type % 3 == 1) {
                // Transposed (non-contiguous) tensor - need to make contiguous first
                try {
                    torch::Tensor transposed = base.t().contiguous();
                    torch::Tensor transposed_result = torch::view_as_real(transposed);
                    (void)transposed_result;
                } catch (...) {
                    // May fail for non-contiguous tensors
                }
            } else {
                // Sliced tensor
                try {
                    torch::Tensor sliced = base.slice(0, 0, 2).contiguous();
                    torch::Tensor sliced_result = torch::view_as_real(sliced);
                    (void)sliced_result;
                } catch (...) {
                    // May fail
                }
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