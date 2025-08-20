#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for rank and dimensions
        if (Size < 2) {
            return 0;
        }
        
        // Parse rank from the first byte
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape for the tensor
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Parse data type for the tensor
        torch::ScalarType dtype = torch::kFloat;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Create options with the parsed dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Test torch.rand with the parsed shape and options
        try {
            torch::Tensor rand_tensor = torch::rand(shape, options);
            
            // Verify the tensor properties
            if (rand_tensor.sizes() != c10::IntArrayRef(shape)) {
                throw std::runtime_error("Tensor shape mismatch");
            }
            
            if (rand_tensor.dtype() != dtype) {
                throw std::runtime_error("Tensor dtype mismatch");
            }
            
            // Check if all values are in the range [0, 1)
            torch::Tensor ge_zero = rand_tensor >= 0;
            torch::Tensor lt_one = rand_tensor < 1;
            torch::Tensor in_range = ge_zero & lt_one;
            bool all_in_range = torch::all(in_range).item<bool>();
            if (!all_in_range) {
                throw std::runtime_error("Values not in range [0, 1)");
            }
            
            // Test edge cases with additional calls if we have more data
            if (offset + 1 < Size) {
                // Try with empty shape
                torch::Tensor empty_shape_tensor = torch::rand({0}, options);
                
                // Try with very large shape if we have enough data
                if (Size > offset + 8) {
                    int64_t large_dim;
                    std::memcpy(&large_dim, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Limit to avoid OOM but still test large dimensions
                    large_dim = std::abs(large_dim) % 1000 + 1;
                    
                    try {
                        torch::Tensor large_tensor = torch::rand({large_dim}, options);
                    } catch (const std::exception& e) {
                        // Expected for very large dimensions
                    }
                }
                
                // Try with negative dimensions (should throw)
                try {
                    torch::Tensor negative_dim_tensor = torch::rand({-1}, options);
                } catch (const std::exception& e) {
                    // Expected behavior
                }
            }
        } catch (const std::exception& e) {
            // Expected for invalid inputs, but we want to continue testing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}