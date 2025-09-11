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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract sections parameter from the remaining data
        int64_t sections = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&sections, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure sections is not zero (would cause division by zero)
            if (sections == 0) {
                sections = 1;
            }
        }
        
        // Extract axis parameter from the remaining data
        int64_t axis = 2; // Default to axis 2 (depth axis)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&axis, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply dsplit operation
        // dsplit splits a tensor into multiple tensors along dimension 2 (depth)
        // If input tensor doesn't have dimension 2, this will throw an exception
        // which is expected behavior for the fuzzer to catch
        std::vector<torch::Tensor> result;
        
        // Try both variants of dsplit: with sections and with indices
        if (offset % 2 == 0) {
            // Use sections variant
            result = torch::dsplit(input_tensor, sections);
        } else {
            // Use indices variant - create a list of indices
            std::vector<int64_t> indices;
            
            // Extract indices from remaining data
            int64_t num_indices = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&num_indices, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Limit number of indices to avoid excessive memory usage
                num_indices = std::abs(num_indices) % 10;
                
                for (int64_t i = 0; i < num_indices && offset + sizeof(int64_t) <= Size; i++) {
                    int64_t idx;
                    std::memcpy(&idx, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    indices.push_back(idx);
                }
                
                // Sort indices to ensure they're in ascending order
                std::sort(indices.begin(), indices.end());
            }
            
            if (!indices.empty()) {
                result = torch::dsplit(input_tensor, indices);
            } else {
                // Fallback to sections variant if no indices were extracted
                result = torch::dsplit(input_tensor, sections);
            }
        }
        
        // Verify the result is not empty
        if (!result.empty()) {
            // Access some elements to ensure computation is not optimized away
            auto first_tensor = result[0];
            auto dtype = first_tensor.dtype();
            auto numel = first_tensor.numel();
            
            // Force evaluation of the tensor
            if (numel > 0) {
                auto sum = first_tensor.sum().item<double>();
                volatile double unused = sum; // Prevent optimization
                (void)unused;
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
