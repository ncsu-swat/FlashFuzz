#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

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
        
        // If tensor is empty or scalar, add a dimension to make split work
        if (num_dims == 0) {
            input_tensor = input_tensor.unsqueeze(0);
            num_dims = 1;
        }
        
        // 1. Determine split dimension
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % num_dims;
        }
        
        // Get the size along the split dimension
        int64_t dim_size = input_tensor.size(dim);
        if (dim_size == 0) {
            return 0;
        }
        
        // 2. Determine split size or sections
        bool use_sections = false;
        int64_t split_size = 1;
        
        if (offset < Size) {
            use_sections = (Data[offset++] % 2 == 0);
        }
        
        if (use_sections && offset < Size) {
            // Use sections approach - sections must sum to dim_size
            uint8_t num_sections = (Data[offset++] % 4) + 1;
            std::vector<int64_t> sections;
            
            int64_t remaining = dim_size;
            for (uint8_t i = 0; i < num_sections - 1 && offset < Size && remaining > 1; ++i) {
                int64_t section = (static_cast<int64_t>(Data[offset++]) % (remaining - 1)) + 1;
                sections.push_back(section);
                remaining -= section;
            }
            // Last section takes the remainder
            if (remaining > 0) {
                sections.push_back(remaining);
            }
            
            // Apply torch::split_with_sizes - wrap in inner try-catch for shape mismatches
            try {
                if (!sections.empty()) {
                    std::vector<torch::Tensor> outputs = torch::split_with_sizes(input_tensor, sections, dim);
                    // Access outputs to ensure computation happens
                    for (const auto& t : outputs) {
                        (void)t.numel();
                    }
                }
            } catch (const std::exception&) {
                // Expected failures due to shape constraints - ignore
            }
        } else {
            // Use split_size approach
            if (offset < Size) {
                split_size = (static_cast<int64_t>(Data[offset++]) % std::min(dim_size, (int64_t)16)) + 1;
            }
            
            // Apply torch::split with size
            try {
                std::vector<torch::Tensor> outputs = torch::split(input_tensor, split_size, dim);
                for (const auto& t : outputs) {
                    (void)t.numel();
                }
            } catch (const std::exception&) {
                // Expected failures - ignore
            }
        }
        
        // Try negative dimension
        if (offset + 1 < Size && Data[offset] % 2 == 0) {
            offset++;
            int64_t neg_dim = -1 - (static_cast<int64_t>(Data[offset++]) % num_dims);
            
            try {
                std::vector<torch::Tensor> outputs = torch::split(input_tensor, split_size, neg_dim);
                for (const auto& t : outputs) {
                    (void)t.numel();
                }
            } catch (const std::exception&) {
                // Expected failures with negative dims - ignore
            }
        }
        
        // Try with split size equal to dim size (single chunk)
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                std::vector<torch::Tensor> outputs = torch::split(input_tensor, dim_size, dim);
                for (const auto& t : outputs) {
                    (void)t.numel();
                }
            } catch (const std::exception&) {
                // ignore
            }
        }
        
        // Try with split size of 1 (maximum chunks)
        if (offset < Size && Data[offset++] % 4 == 0) {
            try {
                std::vector<torch::Tensor> outputs = torch::split(input_tensor, 1, dim);
                for (const auto& t : outputs) {
                    (void)t.numel();
                }
            } catch (const std::exception&) {
                // ignore
            }
        }
        
        // Try splitting along different dimensions
        if (offset < Size && num_dims > 1) {
            int64_t other_dim = (dim + 1) % num_dims;
            int64_t other_dim_size = input_tensor.size(other_dim);
            if (other_dim_size > 0) {
                int64_t other_split = (static_cast<int64_t>(Data[offset++]) % other_dim_size) + 1;
                try {
                    std::vector<torch::Tensor> outputs = torch::split(input_tensor, other_split, other_dim);
                    for (const auto& t : outputs) {
                        (void)t.numel();
                    }
                } catch (const std::exception&) {
                    // ignore
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}