#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip empty inputs
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if the tensor is complex type
        bool is_complex = input_tensor.is_complex();
        
        // If not complex, convert to complex tensor
        if (!is_complex) {
            // Create a complex tensor from the real tensor
            if (input_tensor.dtype() == torch::kFloat) {
                input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
            } else if (input_tensor.dtype() == torch::kDouble) {
                input_tensor = torch::complex(input_tensor, torch::zeros_like(input_tensor));
            } else {
                // For other types, convert to float first
                auto float_tensor = input_tensor.to(torch::kFloat);
                input_tensor = torch::complex(float_tensor, torch::zeros_like(float_tensor));
            }
        }
        
        // Apply torch.imag operation
        torch::Tensor imag_tensor = torch::imag(input_tensor);
        
        // Try some edge cases with different shapes
        if (offset + 1 < Size) {
            // Create a scalar complex tensor
            torch::Tensor scalar_complex = torch::complex(torch::tensor(1.0f), torch::tensor(2.0f));
            torch::Tensor scalar_imag = torch::imag(scalar_complex);
            
            // Create an empty complex tensor
            torch::Tensor empty_complex = torch::complex(torch::empty({0}), torch::empty({0}));
            torch::Tensor empty_imag = torch::imag(empty_complex);
            
            // Create a tensor with extreme values
            double extreme_val = std::numeric_limits<double>::max() / 2.0;
            torch::Tensor extreme_complex = torch::complex(
                torch::ones({1, 1}), 
                torch::full({1, 1}, extreme_val)
            );
            torch::Tensor extreme_imag = torch::imag(extreme_complex);
            
            // Create a tensor with NaN and Inf values
            torch::Tensor special_complex = torch::complex(
                torch::tensor({{1.0, std::numeric_limits<double>::quiet_NaN()}}),
                torch::tensor({{std::numeric_limits<double>::infinity(), 3.0}})
            );
            torch::Tensor special_imag = torch::imag(special_complex);
        }
        
        // Try with a view tensor if possible
        if (!input_tensor.sizes().empty() && input_tensor.numel() > 1) {
            torch::Tensor view_tensor = input_tensor.view({-1});
            torch::Tensor view_imag = torch::imag(view_tensor);
        }
        
        // Try with non-contiguous tensor if possible
        if (input_tensor.dim() > 1 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
            torch::Tensor transposed = input_tensor.transpose(0, 1);
            if (!transposed.is_contiguous()) {
                torch::Tensor transposed_imag = torch::imag(transposed);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
