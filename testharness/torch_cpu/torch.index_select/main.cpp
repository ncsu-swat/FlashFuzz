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
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to select along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create index tensor
        torch::Tensor index_tensor;
        
        // Decide how to create the index tensor
        if (offset < Size) {
            uint8_t index_type = Data[offset++];
            
            // If tensor has dimensions, select a valid dimension
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                
                // Create indices based on the dimension size
                int64_t dim_size = input_tensor.size(dim);
                
                if (index_type % 3 == 0) {
                    // Create a tensor with a single index
                    int64_t idx = 0;
                    if (offset + sizeof(int64_t) <= Size) {
                        std::memcpy(&idx, Data + offset, sizeof(int64_t));
                        offset += sizeof(int64_t);
                    }
                    
                    // Let PyTorch handle out-of-bounds indices
                    index_tensor = torch::tensor({idx}, torch::kInt64);
                }
                else if (index_type % 3 == 1) {
                    // Create a tensor with multiple indices
                    std::vector<int64_t> indices;
                    int64_t num_indices = 1 + (index_type % 5); // 1-5 indices
                    
                    for (int64_t i = 0; i < num_indices; i++) {
                        int64_t idx = 0;
                        if (offset + sizeof(int64_t) <= Size) {
                            std::memcpy(&idx, Data + offset, sizeof(int64_t));
                            offset += sizeof(int64_t);
                        }
                        indices.push_back(idx);
                    }
                    
                    index_tensor = torch::tensor(indices, torch::kInt64);
                }
                else {
                    // Create an empty index tensor
                    index_tensor = torch::tensor({}, torch::kInt64);
                }
            }
            else {
                // For scalar tensors, use a simple index
                index_tensor = torch::tensor({0}, torch::kInt64);
            }
        }
        else {
            // Default index if we don't have enough data
            index_tensor = torch::tensor({0}, torch::kInt64);
        }
        
        // Apply index_select operation
        torch::Tensor result = torch::index_select(input_tensor, dim, index_tensor);
        
        // Perform some operations on the result to ensure it's used
        auto sum = result.sum().item<float>();
        
        // Try another variant with named parameters
        if (input_tensor.dim() > 0) {
            result = torch::index_select(input_tensor, dim, index_tensor);
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
