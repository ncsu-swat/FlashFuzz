#include "fuzzer_utils.h" // General fuzzing utilities
#include <cstdlib>
#include <cstring>
#include <iostream> // For cerr
#include <limits>
#include <optional>
#include <string_view>
#include <tuple> // For std::get with lu_unpack result

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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for hfft if we have more data
        std::optional<torch::SymInt> n_opt = std::nullopt;
        int64_t dim = -1; // Default dimension
        std::optional<std::string_view> norm = std::nullopt;
        
        // Parse n parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_n;
            std::memcpy(&raw_n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);

            // Bound n to a reasonable size to avoid huge allocations
            constexpr int64_t kMaxLength = 4096;
            int64_t abs_n = (raw_n == std::numeric_limits<int64_t>::min())
                                ? std::numeric_limits<int64_t>::max()
                                : std::abs(raw_n);
            if (abs_n > 0)
            {
                int64_t bounded_n = 1 + (abs_n % kMaxLength);
                n_opt = torch::SymInt(bounded_n);
            }
        }
        
        // Parse dim parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If input is not empty tensor, ensure dim is valid
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) dim += input.dim();
            }
        } else if (input.dim() > 0) {
            // Default to last dimension if not specified
            dim = input.dim() - 1;
        }
        
        // Parse norm parameter if we have enough data
        if (offset < Size) {
            bool use_norm = Data[offset] & 0x1;  // Use lowest bit to determine norm
            if (use_norm) {
                uint8_t norm_type = (Data[offset] >> 1) & 0x3;  // Use next 2 bits for norm type
                switch (norm_type) {
                    case 0:
                        norm = "forward";
                        break;
                    case 1:
                        norm = "backward";
                        break;
                    case 2:
                        norm = "ortho";
                        break;
                    default:
                        norm = std::nullopt;
                        break;
                }
            }
            ++offset;
        }
        
        // Apply hfft operation
        int64_t target_dim = dim;
        if (input.dim() == 0) {
            target_dim = -1;
        }
        torch::Tensor output = torch::fft::hfft(input, n_opt, target_dim, norm);
        
        // Force evaluation of the output tensor
        if (output.defined()) {
            volatile float sum = output.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
