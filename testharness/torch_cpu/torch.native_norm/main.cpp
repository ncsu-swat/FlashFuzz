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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for native_norm
        // We need at least 2 more bytes for p and dim
        if (offset + 2 <= Size) {
            // Extract p value (norm type)
            double p;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&p, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else {
                // Not enough data for double, use a byte to determine p
                uint8_t p_selector = Data[offset++];
                // Map to common p values: 0, 1, 2, inf, -inf, or fractional
                switch (p_selector % 6) {
                    case 0: p = 0.0; break;
                    case 1: p = 1.0; break;
                    case 2: p = 2.0; break;
                    case 3: p = INFINITY; break;
                    case 4: p = -INFINITY; break;
                    case 5: p = 0.5 + (p_selector % 10) / 10.0; break;
                }
            }
            
            // Extract dim value
            int64_t dim = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else if (offset < Size) {
                // Use a single byte if not enough data
                dim = static_cast<int64_t>(Data[offset++]);
            }
            
            // Extract keepdim flag
            bool keepdim = false;
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
            
            // Extract dtype option
            c10::optional<torch::ScalarType> dtype = c10::nullopt;
            if (offset < Size) {
                uint8_t dtype_selector = Data[offset++];
                if (dtype_selector & 0x1) {
                    dtype = fuzzer_utils::parseDataType(dtype_selector >> 1);
                }
            }
            
            // Call native_norm with different parameter combinations
            try {
                // Call with all parameters
                torch::Tensor result1 = torch::native_norm(input, c10::optional<torch::Scalar>(p), {dim}, keepdim, dtype);
                
                // Call with only p parameter
                torch::Tensor result2 = torch::native_norm(input, p);
                
                // Call with default p (2.0)
                torch::Tensor result3 = torch::native_norm(input);
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected and handled
            }
        } else {
            // Not enough data for parameters, try with defaults
            try {
                torch::Tensor result = torch::native_norm(input);
            } catch (const c10::Error& e) {
                // PyTorch specific errors are expected and handled
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
