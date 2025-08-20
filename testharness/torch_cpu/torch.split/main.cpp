#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and split parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have some data left for split parameters
        if (offset >= Size) {
            return 0;
        }
        
        // Get tensor dimensions
        int64_t num_dims = input_tensor.dim();
        
        // If tensor is empty or scalar, try with a default dimension
        if (num_dims == 0) {
            // For scalar tensors, we'll add a dimension to make split work
            input_tensor = input_tensor.unsqueeze(0);
            num_dims = 1;
        }
        
        // Extract split parameters from remaining data
        
        // 1. Determine split dimension
        int64_t dim = 0;
        if (offset < Size) {
            // Use the next byte to determine dimension
            dim = static_cast<int64_t>(Data[offset++]) % std::max(1L, num_dims);
        }
        
        // 2. Determine split size or sections
        bool use_sections = false;
        int64_t split_size = 1;
        std::vector<int64_t> sections;
        
        if (offset < Size) {
            // Use the next byte to decide between split_size and sections
            use_sections = (Data[offset++] % 2 == 0);
        }
        
        if (use_sections) {
            // Use sections approach
            // Determine number of sections (1-4)
            uint8_t num_sections = 1;
            if (offset < Size) {
                num_sections = (Data[offset++] % 4) + 1;
            }
            
            // Get section sizes
            for (uint8_t i = 0; i < num_sections && offset < Size; ++i) {
                int64_t section = static_cast<int64_t>(Data[offset++]) + 1; // Ensure positive
                sections.push_back(section);
            }
            
            // Apply torch::split with sections
            std::vector<torch::Tensor> outputs;
            if (!sections.empty()) {
                outputs = torch::split_with_sizes(input_tensor, sections, dim);
            }
        } else {
            // Use split_size approach
            if (offset < Size) {
                // Get a positive split size
                split_size = (static_cast<int64_t>(Data[offset++]) % 16) + 1;
            }
            
            // Apply torch::split with size
            std::vector<torch::Tensor> outputs = torch::split(input_tensor, split_size, dim);
        }
        
        // Try negative dimension
        if (offset < Size && Data[offset++] % 2 == 0) {
            int64_t neg_dim = -1;
            if (num_dims > 0) {
                neg_dim = -1 * (Data[offset++] % num_dims + 1);
            }
            
            // Apply torch::split with negative dimension
            std::vector<torch::Tensor> outputs = torch::split(input_tensor, split_size, neg_dim);
        }
        
        // Try with very large split size
        if (offset < Size && Data[offset++] % 4 == 0) {
            int64_t large_split = std::numeric_limits<int16_t>::max();
            std::vector<torch::Tensor> outputs = torch::split(input_tensor, large_split, dim);
        }
        
        // Try with very small split size
        if (offset < Size && Data[offset++] % 4 == 0) {
            int64_t small_split = 1;
            std::vector<torch::Tensor> outputs = torch::split(input_tensor, small_split, dim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}