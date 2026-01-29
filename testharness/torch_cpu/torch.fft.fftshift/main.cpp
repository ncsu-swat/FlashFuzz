#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <vector>         // For std::vector

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Create input tensor for fftshift
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Parse a byte to determine which variant to test
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++];
        }
        
        torch::Tensor result;
        
        if (variant % 3 == 0) {
            // Variant 1: fftshift without dimension parameter (all dimensions)
            result = torch::fft::fftshift(input_tensor);
        } else if (variant % 3 == 1) {
            // Variant 2: fftshift with single dimension
            int64_t dim = 0;
            if (offset + sizeof(int8_t) <= Size) {
                int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                // Map to valid dimension range
                if (input_tensor.dim() > 0) {
                    dim = dim_byte % input_tensor.dim();
                    // Handle negative modulo
                    if (dim < 0) dim += input_tensor.dim();
                }
            }
            result = torch::fft::fftshift(input_tensor, dim);
        } else {
            // Variant 3: fftshift with multiple dimensions
            std::vector<int64_t> dims;
            if (offset < Size && input_tensor.dim() > 0) {
                uint8_t num_dims = (Data[offset++] % input_tensor.dim()) + 1;
                for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                    int64_t d = Data[offset++] % input_tensor.dim();
                    dims.push_back(d);
                }
            }
            if (dims.empty() && input_tensor.dim() > 0) {
                dims.push_back(0);
            }
            if (!dims.empty()) {
                result = torch::fft::fftshift(input_tensor, dims);
            } else {
                result = torch::fft::fftshift(input_tensor);
            }
        }
        
        // Test ifftshift as the inverse operation
        if (result.defined()) {
            torch::Tensor ifft_result;
            
            if (variant % 3 == 0) {
                ifft_result = torch::fft::ifftshift(result);
            } else if (variant % 3 == 1) {
                int64_t dim = 0;
                if (input_tensor.dim() > 0) {
                    dim = variant % input_tensor.dim();
                }
                ifft_result = torch::fft::ifftshift(result, dim);
            } else {
                // Use all dimensions for ifftshift
                ifft_result = torch::fft::ifftshift(result);
            }
            
            // Access the result to ensure computation happens
            if (ifft_result.defined() && ifft_result.numel() > 0) {
                (void)ifft_result.sum().item<float>();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}