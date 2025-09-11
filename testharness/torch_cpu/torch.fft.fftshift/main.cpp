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
        
        // Create input tensor for fftshift
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have more data to parse a dimension parameter
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply fftshift operation
        torch::Tensor result;
        
        // Try different variants of fftshift
        if (dim >= 0 && dim < input_tensor.dim()) {
            // Apply fftshift with specific dimension
            result = torch::fft::fftshift(input_tensor, dim);
        } else {
            // Apply fftshift without dimension parameter
            result = torch::fft::fftshift(input_tensor);
        }
        
        // Try ifftshift as well if we have a valid result
        if (result.defined()) {
            torch::Tensor ifft_result;
            
            if (dim >= 0 && dim < input_tensor.dim()) {
                ifft_result = torch::fft::ifftshift(result, dim);
            } else {
                ifft_result = torch::fft::ifftshift(result);
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
