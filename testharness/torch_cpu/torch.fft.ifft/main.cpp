#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse FFT dimension parameter if we have more data
        c10::optional<int64_t> dim = c10::nullopt;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t dim_val;
            std::memcpy(&dim_val, Data + offset, sizeof(int64_t));
            dim = dim_val;
            offset += sizeof(int64_t);
        }
        
        // Parse norm parameter if we have more data
        c10::optional<c10::string_view> norm = c10::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 3) {
                case 0: norm = "backward"; break;
                case 1: norm = "ortho"; break;
                case 2: norm = "forward"; break;
            }
        }
        
        // Apply ifft operation
        torch::Tensor result = torch::fft::ifft(input_tensor, c10::nullopt, dim, norm);
        
        // Optionally perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Try with n parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t n;
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Apply ifft with n parameter
            torch::Tensor result_with_n = torch::fft::ifft(input_tensor, c10::optional<int64_t>(n), dim, norm);
            
            // Use the result
            auto sum_with_n = result_with_n.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}