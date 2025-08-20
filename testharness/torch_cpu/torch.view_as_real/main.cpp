#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a complex tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if the tensor is complex
        bool is_complex = input_tensor.is_complex();
        
        // If the tensor is not complex, try to convert it to complex
        if (!is_complex) {
            if (input_tensor.dtype() == torch::kFloat) {
                input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
            } else if (input_tensor.dtype() == torch::kDouble) {
                input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
            } else {
                // Convert to float first, then to complex
                auto float_tensor = input_tensor.to(torch::kFloat);
                input_tensor = torch::complex(float_tensor, torch::zeros_like(float_tensor));
            }
        }
        
        // Apply view_as_real operation
        torch::Tensor result = torch::view_as_real(input_tensor);
        
        // Try to access elements to ensure the operation completed
        if (result.numel() > 0) {
            auto item = result.item();
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
            
            // Create an empty complex tensor
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_complex = torch::empty(empty_shape, torch::kComplexFloat);
            torch::Tensor empty_result = torch::view_as_real(empty_complex);
            
            // Create a tensor with a mix of large and small dimensions
            if (offset + 4 < Size) {
                std::vector<int64_t> mixed_shape;
                for (size_t i = 0; i < 2 && offset + i < Size; i++) {
                    mixed_shape.push_back(Data[offset + i] % 5);  // Small dimensions
                }
                for (size_t i = 2; i < 4 && offset + i < Size; i++) {
                    mixed_shape.push_back(1 + (Data[offset + i] % 3));  // Medium dimensions
                }
                
                if (!mixed_shape.empty()) {
                    torch::Tensor mixed_complex = torch::empty(mixed_shape, torch::kComplexDouble);
                    torch::Tensor mixed_result = torch::view_as_real(mixed_complex);
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
            torch::Tensor complex_tensor = torch::empty({2, 3}, dtype);
            torch::Tensor real_view = torch::view_as_real(complex_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}