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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get sections parameter from the input data
        int64_t sections = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&sections, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure sections is not zero (would cause division by zero)
            if (sections == 0) {
                sections = 1;
            }
        }
        
        // Get axis parameter from the input data
        int64_t axis = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&axis, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply vsplit operation
        // vsplit splits along the first axis (axis=0)
        // If input tensor has less than 2 dimensions, this will throw an exception
        // which is expected behavior for the fuzzer to catch
        std::vector<torch::Tensor> result;
        
        // Try both variants of vsplit: with sections as integer and with sections as array
        if (input_tensor.dim() >= 2) {
            if (offset % 2 == 0) {
                // Use vsplit with integer sections
                result = torch::vsplit(input_tensor, sections);
            } else {
                // Use vsplit with array of indices
                std::vector<int64_t> indices;
                int64_t num_indices = 0;
                
                // Extract indices from remaining data
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&num_indices, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Limit number of indices to avoid excessive memory usage
                    num_indices = std::abs(num_indices) % 10 + 1;
                    
                    for (int64_t i = 0; i < num_indices && offset + sizeof(int64_t) <= Size; i++) {
                        int64_t idx;
                        std::memcpy(&idx, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                        
                        // Ensure indices are within tensor bounds
                        if (idx > 0 && idx < input_tensor.size(0)) {
                            indices.push_back(idx);
                        }
                    }
                    
                    // Sort indices to ensure they're in ascending order
                    if (!indices.empty()) {
                        std::sort(indices.begin(), indices.end());
                        result = torch::vsplit(input_tensor, indices);
                    } else {
                        // Fallback to integer sections if no valid indices
                        result = torch::vsplit(input_tensor, sections);
                    }
                } else {
                    // Not enough data for indices, use integer sections
                    result = torch::vsplit(input_tensor, sections);
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
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}