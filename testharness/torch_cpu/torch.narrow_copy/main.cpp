#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 12) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for narrow_copy from the remaining data
        if (offset + 3 >= Size) {
            return 0;
        }
        
        // Get dimension to narrow along
        int64_t dim = static_cast<int64_t>(Data[offset++]);
        if (input.dim() > 0) {
            dim = dim % input.dim();
        } else {
            // For 0-dim tensor, narrow_copy will fail - let PyTorch handle it
            dim = 0;
        }
        
        // Get start position - constrain to reasonable range for better coverage
        int64_t start = 0;
        if (offset + 2 <= Size) {
            uint16_t start_raw;
            std::memcpy(&start_raw, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            
            // Constrain start to tensor size if possible
            if (input.dim() > 0 && input.size(dim) > 0) {
                start = static_cast<int64_t>(start_raw) % (input.size(dim) + 1);
            } else {
                start = static_cast<int64_t>(start_raw % 16);
            }
        }
        
        // Get length to narrow - constrain to reasonable range
        int64_t length = 1;
        if (offset + 2 <= Size) {
            uint16_t length_raw;
            std::memcpy(&length_raw, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
            
            // Constrain length to remaining size in dimension
            if (input.dim() > 0 && input.size(dim) > start) {
                int64_t max_length = input.size(dim) - start;
                length = 1 + static_cast<int64_t>(length_raw) % max_length;
            } else {
                length = 1 + static_cast<int64_t>(length_raw % 8);
            }
        }
        
        // Apply narrow_copy operation
        torch::Tensor result = torch::narrow_copy(input, dim, start, length);
        
        // Basic sanity check to ensure the result is used
        if (result.defined()) {
            volatile auto num_elements = result.numel();
            (void)num_elements;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}