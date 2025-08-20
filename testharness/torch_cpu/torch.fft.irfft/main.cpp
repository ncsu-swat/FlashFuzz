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
        
        // Extract parameters for irfft
        int64_t n = 0;
        int64_t dim = -1;
        std::string norm = "backward";
        
        // Parse n parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Parse norm parameter if we have enough data
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++];
            switch (norm_selector % 3) {
                case 0:
                    norm = "backward";
                    break;
                case 1:
                    norm = "forward";
                    break;
                case 2:
                    norm = "ortho";
                    break;
            }
        }
        
        // Apply irfft operation with different parameter combinations
        torch::Tensor output;
        
        // Try different combinations of parameters
        if (n > 0) {
            output = torch::fft::irfft(input, n, dim, norm);
        } else {
            output = torch::fft::irfft(input, c10::nullopt, dim, norm);
        }
        
        // Ensure the output is used to prevent optimization
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