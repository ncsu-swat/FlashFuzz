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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create real and imaginary tensors
        torch::Tensor real = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the imaginary part
        if (offset >= Size) {
            // Try with a simple operation anyway
            torch::Tensor imag = real.clone();
            torch::Tensor complex_tensor = torch::complex(real, imag);
            return 0;
        }
        
        torch::Tensor imag = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to make the tensors compatible if they have different shapes
        if (real.sizes() != imag.sizes()) {
            // Attempt to broadcast or reshape to make them compatible
            // Option 1: Try to use the smaller shape for both
            if (real.numel() < imag.numel()) {
                try {
                    imag = imag.reshape_as(real);
                } catch (...) {
                    // If reshape fails, try a different approach
                    try {
                        real = real.expand_as(imag);
                    } catch (...) {
                        // If both approaches fail, create a new tensor with compatible shape
                        imag = real.clone();
                    }
                }
            } else {
                try {
                    real = real.reshape_as(imag);
                } catch (...) {
                    try {
                        imag = imag.expand_as(real);
                    } catch (...) {
                        real = imag.clone();
                    }
                }
            }
        }
        
        // Ensure compatible dtypes for complex creation
        // Complex requires floating point inputs
        if (!real.is_floating_point()) {
            real = real.to(torch::kFloat);
        }
        if (!imag.is_floating_point()) {
            imag = imag.to(torch::kFloat);
        }
        
        // Apply the torch.complex operation
        torch::Tensor complex_tensor = torch::complex(real, imag);
        
        // Test some operations on the complex tensor
        if (complex_tensor.numel() > 0) {
            torch::Tensor abs_val = torch::abs(complex_tensor);
            torch::Tensor angle_val = torch::angle(complex_tensor);
            torch::Tensor real_part = torch::real(complex_tensor);
            torch::Tensor imag_part = torch::imag(complex_tensor);
        }
        
        // Try edge cases with scalar inputs
        if (offset + 2 <= Size) {
            double scalar_real = static_cast<double>(Data[offset++]) / 255.0;
            double scalar_imag = static_cast<double>(Data[offset++]) / 255.0;
            
            // Test scalar + tensor
            torch::Tensor complex_scalar_tensor = torch::complex(
                torch::tensor(scalar_real), 
                imag
            );
            
            // Test tensor + scalar
            torch::Tensor complex_tensor_scalar = torch::complex(
                real,
                torch::tensor(scalar_imag)
            );
        }
        
        // Try with zero-sized tensors if we have enough data
        if (offset + 1 < Size) {
            std::vector<int64_t> empty_shape = {0};
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            torch::Tensor empty_real = torch::empty(empty_shape, options);
            torch::Tensor empty_imag = torch::empty(empty_shape, options);
            
            torch::Tensor empty_complex = torch::complex(empty_real, empty_imag);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
