#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <optional>
#include <string_view>
#include <cstring>
#include <cstdlib>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
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
        int64_t dim = -1;
        if (offset + sizeof(int64_t) <= Size) {
            int64_t dim_val;
            std::memcpy(&dim_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            auto rank = input_tensor.dim();
            if (rank > 0) {
                // Clamp dim into valid range [-rank, rank-1]
                dim = dim_val % rank;
                if (dim < 0) {
                    dim += rank;
                }
            }
        }
        
        // Parse norm parameter if we have more data
        std::optional<std::string_view> norm = std::nullopt;
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 3) {
                case 0: norm = "backward"; break;
                case 1: norm = "ortho"; break;
                case 2: norm = "forward"; break;
            }
        }

        // Apply ifft operation
        torch::Tensor result = torch::fft::ifft(input_tensor, std::nullopt, dim, norm);
        
        // Optionally perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Try with n parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t n_raw;
            std::memcpy(&n_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);

            // Keep n within a small positive range to avoid huge allocations
            int64_t n = 1 + std::abs(n_raw % 64);
            
            // Apply ifft with n parameter
            std::optional<c10::SymInt> n_opt = c10::SymInt(n);
            torch::Tensor result_with_n = torch::fft::ifft(input_tensor, n_opt, dim, norm);
            
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
