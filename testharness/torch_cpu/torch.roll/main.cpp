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
        
        // Parse shift amount
        int64_t shifts = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&shifts, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dimensions
        std::vector<int64_t> dims;
        uint8_t num_dims = 0;
        if (offset < Size) {
            num_dims = Data[offset++] % 4; // Limit to a reasonable number of dimensions
            
            for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t dim;
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dims.push_back(dim);
            }
        }
        
        // Test different variants of torch::roll
        
        // Case 1: Roll with shifts only (no dims)
        torch::Tensor result1 = torch::roll(input_tensor, shifts);
        
        // Case 2: Roll with shifts and single dimension
        if (!dims.empty()) {
            torch::Tensor result2 = torch::roll(input_tensor, shifts, dims[0]);
        }
        
        // Case 3: Roll with shifts and multiple dimensions
        if (dims.size() > 1) {
            // Create shifts vector with same number of elements as dims
            std::vector<int64_t> shifts_vec(dims.size(), shifts);
            
            torch::Tensor result3 = torch::roll(input_tensor, shifts_vec, dims);
        }
        
        // Case 4: Roll with negative shifts
        torch::Tensor result4 = torch::roll(input_tensor, -shifts);
        
        // Case 5: Roll with out-of-bounds dimension
        if (input_tensor.dim() > 0) {
            int64_t out_of_bounds_dim = input_tensor.dim() + 5;
            try {
                torch::Tensor result5 = torch::roll(input_tensor, shifts, out_of_bounds_dim);
            } catch (const c10::Error &) {
                // Expected exception for out-of-bounds dimension
            }
        }
        
        // Case 6: Roll with negative dimension
        try {
            torch::Tensor result6 = torch::roll(input_tensor, shifts, -1);
        } catch (const c10::Error &) {
            // May throw depending on implementation
        }
        
        // Case 7: Roll with zero shifts
        torch::Tensor result7 = torch::roll(input_tensor, 0);
        
        // Case 8: Roll with very large shifts
        int64_t large_shift = std::numeric_limits<int32_t>::max();
        torch::Tensor result8 = torch::roll(input_tensor, large_shift);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
