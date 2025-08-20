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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for hfft if we have more data
        int64_t n = -1;  // Default: -1 means use the default size
        int64_t dim = -1; // Default dimension
        std::optional<std::string_view> norm = std::nullopt;
        
        // Parse n parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
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
        }
        
        // Apply hfft operation
        torch::Tensor output;
        
        // Try different parameter combinations
        if (n == -1) {
            if (dim == -1) {
                // Use default n and dim
                output = torch::fft::hfft(input, c10::nullopt, norm);
            } else {
                // Use default n, specified dim
                output = torch::fft::hfft(input, c10::nullopt, dim, norm);
            }
        } else {
            if (dim == -1) {
                // Use specified n, default dim
                output = torch::fft::hfft(input, n, norm);
            } else {
                // Use specified n and dim
                output = torch::fft::hfft(input, n, dim, norm);
            }
        }
        
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