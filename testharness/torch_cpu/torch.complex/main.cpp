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
            // Ensure floating point
            if (!real.is_floating_point()) {
                real = real.to(torch::kFloat);
                imag = imag.to(torch::kFloat);
            }
            torch::Tensor complex_tensor = torch::complex(real, imag);
            return 0;
        }
        
        torch::Tensor imag = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to make the tensors compatible if they have different shapes
        if (real.sizes() != imag.sizes()) {
            // Attempt to broadcast or reshape to make them compatible
            if (real.numel() < imag.numel()) {
                try {
                    imag = imag.reshape_as(real);
                } catch (...) {
                    try {
                        real = real.expand_as(imag);
                    } catch (...) {
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
        // Complex requires floating point inputs of the same type
        if (!real.is_floating_point()) {
            real = real.to(torch::kFloat);
        }
        if (!imag.is_floating_point()) {
            imag = imag.to(torch::kFloat);
        }
        // Ensure both have the same dtype
        if (real.dtype() != imag.dtype()) {
            imag = imag.to(real.dtype());
        }
        
        // Apply the torch.complex operation
        torch::Tensor complex_tensor = torch::complex(real, imag);
        
        // Test some operations on the complex tensor
        if (complex_tensor.numel() > 0) {
            torch::Tensor abs_val = torch::abs(complex_tensor);
            torch::Tensor angle_val = torch::angle(complex_tensor);
            torch::Tensor real_part = torch::real(complex_tensor);
            torch::Tensor imag_part = torch::imag(complex_tensor);
            
            // Additional operations to improve coverage
            torch::Tensor conj_val = torch::conj(complex_tensor);
            torch::Tensor neg_val = -complex_tensor;
        }
        
        // Try edge cases with scalar inputs
        if (offset + 2 <= Size) {
            double scalar_real = static_cast<double>(Data[offset++]) / 255.0;
            double scalar_imag = static_cast<double>(Data[offset++]) / 255.0;
            
            try {
                // Test scalar + tensor (needs compatible shapes via broadcasting)
                torch::Tensor scalar_tensor = torch::tensor(scalar_real, torch::kFloat);
                torch::Tensor complex_scalar_tensor = torch::complex(
                    scalar_tensor.expand_as(imag), 
                    imag
                );
                
                // Test tensor + scalar
                torch::Tensor imag_scalar_tensor = torch::tensor(scalar_imag, torch::kFloat);
                torch::Tensor complex_tensor_scalar = torch::complex(
                    real,
                    imag_scalar_tensor.expand_as(real)
                );
            } catch (...) {
                // Silently ignore broadcasting failures
            }
        }
        
        // Try with zero-sized tensors if we have enough data
        if (offset + 1 < Size) {
            std::vector<int64_t> empty_shape = {0};
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            torch::Tensor empty_real = torch::empty(empty_shape, options);
            torch::Tensor empty_imag = torch::empty(empty_shape, options);
            
            torch::Tensor empty_complex = torch::complex(empty_real, empty_imag);
        }
        
        // Test with different floating point types based on fuzzer data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 3;
            torch::Dtype target_dtype;
            switch (dtype_selector) {
                case 0: target_dtype = torch::kFloat; break;
                case 1: target_dtype = torch::kDouble; break;
                case 2: target_dtype = torch::kHalf; break;
                default: target_dtype = torch::kFloat; break;
            }
            
            try {
                torch::Tensor typed_real = real.to(target_dtype);
                torch::Tensor typed_imag = imag.to(target_dtype);
                torch::Tensor typed_complex = torch::complex(typed_real, typed_imag);
            } catch (...) {
                // Some dtypes may not be supported on all platforms
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}