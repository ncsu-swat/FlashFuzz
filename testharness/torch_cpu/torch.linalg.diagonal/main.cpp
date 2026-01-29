#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // linalg.diagonal requires at least 2 dimensions
        if (input.dim() < 2) {
            // Reshape to 2D if needed
            int64_t numel = input.numel();
            if (numel == 0) {
                return 0;
            }
            int64_t side = static_cast<int64_t>(std::sqrt(static_cast<double>(numel)));
            if (side < 1) side = 1;
            int64_t other = numel / side;
            if (side * other != numel) {
                // Can't reshape evenly, just make it square-ish
                input = input.reshape({-1}).slice(0, 0, side * side).reshape({side, side});
            } else {
                input = input.reshape({side, other});
            }
        }
        
        int64_t ndim = input.dim();
        
        // Extract parameters for diagonal operation
        int64_t offset_val = 0;
        int64_t dim1 = -2;  // Default in PyTorch
        int64_t dim2 = -1;  // Default in PyTorch
        
        // Parse offset parameter if we have more data
        if (offset + sizeof(int8_t) <= Size) {
            // Use smaller type to get reasonable offset values
            int8_t small_offset;
            std::memcpy(&small_offset, Data + offset, sizeof(int8_t));
            offset_val = small_offset;  // Extend to int64_t
            offset += sizeof(int8_t);
        }
        
        // Parse dim1 parameter if we have more data
        if (offset + sizeof(uint8_t) <= Size && ndim > 0) {
            uint8_t dim_byte;
            std::memcpy(&dim_byte, Data + offset, sizeof(uint8_t));
            dim1 = static_cast<int64_t>(dim_byte % ndim);
            offset += sizeof(uint8_t);
        }
        
        // Parse dim2 parameter if we have more data
        if (offset + sizeof(uint8_t) <= Size && ndim > 0) {
            uint8_t dim_byte;
            std::memcpy(&dim_byte, Data + offset, sizeof(uint8_t));
            dim2 = static_cast<int64_t>(dim_byte % ndim);
            offset += sizeof(uint8_t);
        }
        
        // Ensure dim1 != dim2 (required by the API)
        if (dim1 == dim2 && ndim > 1) {
            dim2 = (dim1 + 1) % ndim;
        }
        
        // Apply the linalg.diagonal operation
        torch::Tensor result;
        
        try {
            // Try different variants of the operation based on fuzzer input
            uint8_t variant = (Size > offset) ? Data[Size - 1] % 3 : 0;
            
            if (variant == 0) {
                // Variant 1: Default parameters
                result = torch::linalg_diagonal(input);
            } else if (variant == 1) {
                // Variant 2: Specify offset
                result = torch::linalg_diagonal(input, offset_val);
            } else {
                // Variant 3: Specify all parameters
                result = torch::linalg_diagonal(input, offset_val, dim1, dim2);
            }
        } catch (const c10::Error &e) {
            // Expected errors for invalid parameter combinations
            return 0;
        }
        
        // Perform some operations on the result to ensure it's used
        if (result.defined() && result.numel() > 0) {
            auto sum = result.sum();
            if (sum.defined()) {
                volatile double val = sum.item<double>();
                (void)val;
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