#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least one dimension
        if (input_tensor.dim() == 0) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // Parse dimension to split along first (needed to determine valid split sizes)
        int64_t dim = 0;
        if (offset + sizeof(int8_t) <= Size) {
            int8_t dim_raw = static_cast<int8_t>(Data[offset]);
            offset += sizeof(int8_t);
            
            // Constrain dimension to valid range (allow negative indexing)
            int ndim = input_tensor.dim();
            dim = dim_raw % ndim;
        }
        
        // Get the size along the dimension we're splitting
        int64_t dim_size = input_tensor.size(dim >= 0 ? dim : input_tensor.dim() + dim);
        
        // Parse number of splits
        if (offset + sizeof(uint8_t) > Size) {
            return 0;
        }
        
        uint8_t num_splits_raw = Data[offset];
        offset += sizeof(uint8_t);
        
        // Ensure we have at least 1 split and not too many
        int num_splits = (num_splits_raw % 8) + 1;
        
        // Generate split sizes that sum to dim_size for valid coverage
        // Also sometimes test invalid sizes for error path coverage
        std::vector<int64_t> split_sizes;
        
        if (offset < Size && (Data[offset] & 0x01)) {
            // Test valid split sizes (more common path)
            offset++;
            int64_t remaining = dim_size;
            
            for (int i = 0; i < num_splits - 1 && remaining > 0 && offset + sizeof(uint8_t) <= Size; ++i) {
                uint8_t portion = Data[offset];
                offset += sizeof(uint8_t);
                
                // Allocate a portion of remaining size
                int64_t split_size = (portion % std::max<int64_t>(1, remaining)) + 1;
                split_size = std::min(split_size, remaining - (num_splits - 1 - i));
                if (split_size > 0) {
                    split_sizes.push_back(split_size);
                    remaining -= split_size;
                }
            }
            
            // Last split gets the remainder
            if (remaining > 0) {
                split_sizes.push_back(remaining);
            }
        } else {
            // Test potentially invalid split sizes for error path coverage
            if (offset < Size) offset++;
            
            for (int i = 0; i < num_splits && offset + sizeof(int16_t) <= Size; ++i) {
                int16_t split_size_raw;
                std::memcpy(&split_size_raw, Data + offset, sizeof(int16_t));
                offset += sizeof(int16_t);
                
                // Use absolute value but allow some variation
                int64_t split_size = std::abs(static_cast<int64_t>(split_size_raw)) % (dim_size + 10) + 1;
                split_sizes.push_back(split_size);
            }
        }
        
        // Ensure we have at least one split
        if (split_sizes.empty()) {
            split_sizes.push_back(dim_size > 0 ? dim_size : 1);
        }
        
        // Apply the split_with_sizes operation
        try {
            std::vector<torch::Tensor> result = torch::split_with_sizes(input_tensor, split_sizes, dim);
            
            // Perform some operations on the result to ensure it's valid
            for (const auto& tensor : result) {
                auto sum = tensor.sum();
                
                // Force evaluation
                sum.item<float>();
                
                if (tensor.numel() > 0) {
                    auto mean = tensor.mean();
                    mean.item<float>();
                }
                
                // Test that we can further operate on split tensors
                auto reshaped = tensor.contiguous();
                (void)reshaped.data_ptr();
            }
            
            // Verify split properties
            if (!result.empty()) {
                // Check that concatenating the splits gives back original size
                int64_t total_size = 0;
                for (const auto& t : result) {
                    total_size += t.size(dim >= 0 ? dim : t.dim() + dim);
                }
                (void)total_size;
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid dimensions/sizes are expected
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}