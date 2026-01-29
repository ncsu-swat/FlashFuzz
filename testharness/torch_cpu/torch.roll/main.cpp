#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if tensor is empty or has no dimensions
        if (input_tensor.numel() == 0 || input_tensor.dim() == 0) {
            return 0;
        }
        
        // Parse shift amount (limit to reasonable range)
        int64_t shifts = 0;
        if (offset + sizeof(int32_t) <= Size) {
            int32_t shift_val;
            std::memcpy(&shift_val, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            shifts = shift_val; // Use int32 to avoid extreme values
        }
        
        // Parse dimensions - bound to valid tensor dimensions
        std::vector<int64_t> dims;
        uint8_t num_dims = 0;
        if (offset < Size) {
            num_dims = (Data[offset++] % std::min(static_cast<int64_t>(3), input_tensor.dim())) + 1;
            
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                // Map byte to valid dimension range
                int64_t dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                dims.push_back(dim);
            }
        }
        
        // Case 1: Roll with shifts only (flattens tensor, rolls, then reshapes)
        {
            torch::Tensor result1 = torch::roll(input_tensor, shifts);
            (void)result1;
        }
        
        // Case 2: Roll with shifts and single dimension
        if (!dims.empty()) {
            try {
                torch::Tensor result2 = torch::roll(input_tensor, shifts, dims[0]);
                (void)result2;
            } catch (const c10::Error &) {
                // Silently handle invalid dimension
            }
        }
        
        // Case 3: Roll with shifts vector and dims vector (must be same size)
        if (dims.size() > 1) {
            std::vector<int64_t> shifts_vec(dims.size(), shifts);
            try {
                torch::Tensor result3 = torch::roll(input_tensor, at::IntArrayRef(shifts_vec), at::IntArrayRef(dims));
                (void)result3;
            } catch (const c10::Error &) {
                // Silently handle invalid configuration
            }
        }
        
        // Case 4: Roll with negative shifts
        {
            torch::Tensor result4 = torch::roll(input_tensor, -shifts);
            (void)result4;
        }
        
        // Case 5: Roll with negative dimension index
        if (input_tensor.dim() > 0) {
            try {
                torch::Tensor result5 = torch::roll(input_tensor, shifts, -1);
                (void)result5;
            } catch (const c10::Error &) {
                // Silently handle if negative indexing fails
            }
        }
        
        // Case 6: Roll with zero shifts
        {
            torch::Tensor result6 = torch::roll(input_tensor, 0);
            (void)result6;
        }
        
        // Case 7: Roll along each valid dimension
        for (int64_t d = 0; d < input_tensor.dim(); ++d) {
            try {
                torch::Tensor result7 = torch::roll(input_tensor, shifts, d);
                (void)result7;
            } catch (const c10::Error &) {
                // Silently handle unexpected errors
            }
        }
        
        // Case 8: Roll with multiple shifts along multiple dims
        if (input_tensor.dim() >= 2) {
            std::vector<int64_t> multi_shifts = {shifts, -shifts};
            std::vector<int64_t> multi_dims = {0, 1};
            try {
                torch::Tensor result8 = torch::roll(input_tensor, at::IntArrayRef(multi_shifts), at::IntArrayRef(multi_dims));
                (void)result8;
            } catch (const c10::Error &) {
                // Silently handle if dimensions don't match
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