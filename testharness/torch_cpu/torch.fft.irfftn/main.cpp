#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dimensions for irfftn
        std::vector<int64_t> dims;
        if (offset + 1 < Size) {
            uint8_t num_dims = Data[offset++] % 5; // Up to 4 dimensions
            
            for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim;
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dims.push_back(dim);
            }
        }
        
        // Parse norm parameter
        std::string norm = "backward";
        if (offset < Size) {
            uint8_t norm_selector = Data[offset++] % 4;
            switch (norm_selector) {
                case 0: norm = "backward"; break;
                case 1: norm = "forward"; break;
                case 2: norm = "ortho"; break;
                default: norm = "backward";
            }
        }
        
        // Parse s parameter (output signal size)
        std::optional<std::vector<int64_t>> s = std::nullopt;
        if (offset < Size && Data[offset++] % 2 == 0) { // 50% chance to use s parameter
            s = std::vector<int64_t>();
            uint8_t s_size = (offset < Size) ? (Data[offset++] % 5) : 0; // Up to 4 dimensions
            
            for (uint8_t i = 0; i < s_size && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim_size;
                std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                s->push_back(dim_size);
            }
        }
        
        // Apply irfftn operation with different parameter combinations
        torch::Tensor result;
        
        if (dims.empty() && !s.has_value()) {
            // Case 1: No dims, no s
            result = torch::fft::irfftn(input, s, c10::nullopt, norm);
        } 
        else if (!dims.empty() && !s.has_value()) {
            // Case 2: With dims, no s
            result = torch::fft::irfftn(input, s, dims, norm);
        }
        else if (dims.empty() && s.has_value()) {
            // Case 3: No dims, with s
            result = torch::fft::irfftn(input, s, c10::nullopt, norm);
        }
        else {
            // Case 4: With dims, with s
            result = torch::fft::irfftn(input, s, dims, norm);
        }
        
        // Access result to ensure computation is performed
        auto sum = result.sum().item<double>();
        (void)sum; // Prevent unused variable warning
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}