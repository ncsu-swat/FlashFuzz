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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply ihfft operation
        // ihfft expects a real-valued input tensor
        // If the input tensor is complex, convert it to real
        if (input.is_complex()) {
            input = torch::real(input);
        }
        
        // Get a dimension to apply ihfft along
        int64_t dim = -1;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % (input.dim() + 1) - 1;
        }
        
        // Get norm parameter
        c10::string_view norm = "backward";
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 3;
            if (norm_selector == 0) {
                norm = "backward";
            } else if (norm_selector == 1) {
                norm = "forward";
            } else {
                norm = "ortho";
            }
        }
        
        // Apply ihfft operation
        torch::Tensor result;
        if (input.dim() > 0) {
            if (dim >= 0 && dim < input.dim()) {
                // Apply ihfft along the specified dimension
                result = torch::fft::ihfft(input, c10::nullopt, dim, norm);
            } else {
                // Apply ihfft along the last dimension
                result = torch::fft::ihfft(input, c10::nullopt, -1, norm);
            }
        } else {
            // For 0-dim tensors, just apply ihfft without dimension
            result = torch::fft::ihfft(input, c10::nullopt, c10::nullopt, norm);
        }
        
        // Try with n parameter (length of transformed axis)
        if (offset + sizeof(int64_t) <= Size) {
            int64_t n_raw;
            std::memcpy(&n_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make n positive and reasonable
            int64_t n = std::abs(n_raw) % 100 + 1;
            
            // Apply ihfft with n parameter
            if (input.dim() > 0) {
                if (dim >= 0 && dim < input.dim()) {
                    result = torch::fft::ihfft(input, c10::optional<int64_t>(n), dim, norm);
                } else {
                    result = torch::fft::ihfft(input, c10::optional<int64_t>(n), -1, norm);
                }
            } else {
                result = torch::fft::ihfft(input, c10::optional<int64_t>(n), c10::nullopt, norm);
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
