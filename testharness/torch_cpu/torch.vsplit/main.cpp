#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::sort

// --- Fuzzer Entry Point ---
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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // vsplit requires at least 2 dimensions
        if (input_tensor.dim() < 2) {
            return 0;
        }
        
        // Get the size along the first axis
        int64_t dim0_size = input_tensor.size(0);
        if (dim0_size == 0) {
            return 0;
        }
        
        // Get sections parameter from the input data
        int64_t sections = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&sections, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure sections is positive and reasonable
            sections = std::abs(sections);
            if (sections == 0) {
                sections = 1;
            }
            // Limit sections to avoid excessive splits
            sections = (sections % 16) + 1;
        }
        
        std::vector<torch::Tensor> result;
        
        // Decide which variant to use based on fuzzer data
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++];
        }
        
        if (variant % 2 == 0) {
            // Use vsplit with integer sections
            // sections must evenly divide dim0_size
            // Find a valid divisor
            int64_t valid_sections = 1;
            for (int64_t s = sections; s >= 1; s--) {
                if (dim0_size % s == 0) {
                    valid_sections = s;
                    break;
                }
            }
            
            try {
                result = torch::vsplit(input_tensor, valid_sections);
            } catch (...) {
                // Silently handle expected failures
            }
        } else {
            // Use vsplit with array of indices (split points)
            std::vector<int64_t> indices;
            int64_t num_indices = 0;
            
            // Extract indices from remaining data
            if (offset + sizeof(int8_t) <= Size) {
                num_indices = static_cast<int8_t>(Data[offset++]);
                
                // Limit number of indices to avoid excessive memory usage
                num_indices = std::abs(num_indices) % 10 + 1;
                
                for (int64_t i = 0; i < num_indices && offset + sizeof(int16_t) <= Size; i++) {
                    int16_t idx_raw;
                    std::memcpy(&idx_raw, Data + offset, sizeof(int16_t));
                    offset += sizeof(int16_t);
                    
                    // Map to valid range within tensor bounds (exclusive of 0 and dim0_size)
                    int64_t idx = (std::abs(idx_raw) % (dim0_size > 1 ? dim0_size - 1 : 1)) + 1;
                    if (idx > 0 && idx < dim0_size) {
                        indices.push_back(idx);
                    }
                }
                
                if (!indices.empty()) {
                    // Sort and remove duplicates
                    std::sort(indices.begin(), indices.end());
                    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
                    
                    try {
                        result = torch::vsplit(input_tensor, indices);
                    } catch (...) {
                        // Silently handle expected failures
                    }
                }
            }
            
            // Fallback to single section if no valid indices
            if (result.empty()) {
                try {
                    result = torch::vsplit(input_tensor, 1);
                } catch (...) {
                    // Silently handle expected failures
                }
            }
        }
        
        // Verify the results by checking properties
        for (const auto& tensor : result) {
            // Access tensor properties to ensure they're valid
            auto sizes = tensor.sizes();
            auto dtype = tensor.dtype();
            auto numel = tensor.numel();
            
            // Perform a simple operation on each result tensor
            // to ensure they're valid and usable
            if (numel > 0) {
                auto sum = tensor.sum();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}