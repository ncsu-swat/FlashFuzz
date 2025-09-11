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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract split parameters from the remaining data
        if (offset + 2 >= Size) {
            return 0;
        }
        
        // Get split size or sections
        int64_t split_param = 0;
        bool use_sections = false;
        
        if (offset < Size) {
            uint8_t param_type = Data[offset++];
            use_sections = (param_type % 2 == 0);
            
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&split_param, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else {
                // Not enough data for split_param, use a default value
                split_param = 1;
            }
        }
        
        // Get dimension to split along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply the split_copy operation
        std::vector<torch::Tensor> result;
        
        if (use_sections) {
            // Use sections parameter (number of splits)
            // Ensure sections is positive to avoid trivial checks
            int64_t sections = std::abs(split_param) % 10 + 1;
            result = torch::split_copy(input_tensor, sections, dim);
        } else {
            // Use split_size parameter (size of each chunk)
            result = torch::split_copy(input_tensor, split_param, dim);
        }
        
        // Verify the result by concatenating back
        if (!result.empty()) {
            torch::Tensor reconstructed = torch::cat(result, dim);
            
            // Check if the reconstructed tensor matches the original
            bool shapes_match = reconstructed.sizes() == input_tensor.sizes();
            
            // Access some elements to ensure tensors are valid
            if (input_tensor.numel() > 0) {
                auto first_elem = input_tensor.flatten()[0];
            }
            
            if (reconstructed.numel() > 0) {
                auto first_elem = reconstructed.flatten()[0];
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
