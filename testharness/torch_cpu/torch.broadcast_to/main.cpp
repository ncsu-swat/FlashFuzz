#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse target shape for broadcast_to
        // Need at least 1 byte for rank
        if (offset >= Size) {
            return 0;
        }
        
        // Parse rank for target shape
        uint8_t target_rank_byte = Data[offset++];
        uint8_t target_rank = fuzzer_utils::parseRank(target_rank_byte);
        
        // Parse target shape
        std::vector<int64_t> target_shape;
        if (offset < Size) {
            target_shape = fuzzer_utils::parseShape(Data, offset, Size, target_rank);
        } else {
            // If we don't have enough data, create a simple shape
            for (uint8_t i = 0; i < target_rank; ++i) {
                target_shape.push_back(1 + (i % 5)); // Some arbitrary dimensions
            }
        }
        
        // Apply broadcast_to operation
        torch::Tensor result = torch::broadcast_to(input_tensor, target_shape);
        
        // Verify the result has the expected shape
        auto result_sizes = result.sizes();
        for (size_t i = 0; i < target_shape.size(); ++i) {
            if (i < result_sizes.size() && result_sizes[i] != target_shape[i]) {
                throw std::runtime_error("Result shape doesn't match target shape");
            }
        }
        
        // Try some edge cases if we have more data
        if (offset + 1 < Size) {
            // Try broadcasting to a shape with a dimension of 0
            std::vector<int64_t> zero_dim_shape = target_shape;
            if (!zero_dim_shape.empty()) {
                zero_dim_shape[0] = 0;
                try {
                    torch::Tensor zero_result = torch::broadcast_to(input_tensor, zero_dim_shape);
                } catch (const std::exception&) {
                    // Expected exception for invalid broadcasting
                }
            }
            
            // Try broadcasting to a shape with fewer dimensions than the input
            if (input_tensor.dim() > 1) {
                std::vector<int64_t> smaller_shape;
                for (int i = 0; i < input_tensor.dim() - 1; ++i) {
                    smaller_shape.push_back(input_tensor.size(i));
                }
                try {
                    torch::Tensor smaller_result = torch::broadcast_to(input_tensor, smaller_shape);
                } catch (const std::exception&) {
                    // Expected exception for invalid broadcasting
                }
            }
            
            // Try broadcasting to a shape with incompatible dimensions
            std::vector<int64_t> incompatible_shape = target_shape;
            if (!incompatible_shape.empty() && input_tensor.dim() > 0) {
                for (size_t i = 0; i < std::min(incompatible_shape.size(), static_cast<size_t>(input_tensor.dim())); ++i) {
                    if (input_tensor.size(i) > 1) {
                        incompatible_shape[i] = input_tensor.size(i) + 1;
                        break;
                    }
                }
                try {
                    torch::Tensor incompatible_result = torch::broadcast_to(input_tensor, incompatible_shape);
                } catch (const std::exception&) {
                    // Expected exception for invalid broadcasting
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