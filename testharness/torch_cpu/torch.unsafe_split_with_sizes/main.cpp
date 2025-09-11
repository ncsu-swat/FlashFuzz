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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dimension to split along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor is not empty, make sure dim is within valid range
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                if (dim < 0) {
                    dim += input_tensor.dim();
                }
            }
        }
        
        // Parse number of sections
        uint8_t num_sections = 1;
        if (offset < Size) {
            num_sections = Data[offset++];
            // Ensure at least 1 section
            num_sections = std::max(num_sections, static_cast<uint8_t>(1));
            // Limit to a reasonable number to avoid excessive memory usage
            num_sections = std::min(num_sections, static_cast<uint8_t>(16));
        }
        
        // Create section sizes vector
        std::vector<int64_t> section_sizes;
        for (uint8_t i = 0; i < num_sections; ++i) {
            int64_t size_val = 1;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&size_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow negative sizes to test error handling
                // No need to clamp or sanitize - let the operation handle it
            }
            section_sizes.push_back(size_val);
        }
        
        // Apply the unsafe_split_with_sizes operation
        if (input_tensor.dim() > 0) {
            std::vector<torch::Tensor> result = torch::unsafe_split_with_sizes(input_tensor, section_sizes, dim);
            
            // Perform some basic operations on the result to ensure it's used
            if (!result.empty()) {
                torch::Tensor sum = torch::zeros_like(result[0]);
                for (const auto& t : result) {
                    sum = sum + t.sum();
                }
            }
        }
        else {
            // For 0-dim tensors, still try the operation to see how it handles it
            std::vector<torch::Tensor> result = torch::unsafe_split_with_sizes(input_tensor, section_sizes, dim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
