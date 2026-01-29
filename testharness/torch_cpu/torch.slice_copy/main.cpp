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
        
        // Need at least a few bytes to create a tensor and slice parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // slice_copy requires at least 1-dimensional tensor
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Get parameters for slice_copy operation
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Get dimension to slice along
        int64_t dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
        
        // Get size along the dimension for better parameter generation
        int64_t dim_size = input_tensor.size(dim);
        
        // Get start index (relative to dimension size)
        int64_t start = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Normalize to reasonable range
            if (dim_size > 0) {
                start = start % (dim_size + 1);
            }
        }
        
        // Get end index
        int64_t end = dim_size;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Normalize to reasonable range
            if (dim_size > 0) {
                end = end % (dim_size + 1);
            }
        }
        
        // Get step value (must be positive for slice_copy)
        int64_t step = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Ensure step is positive and reasonable
            step = std::abs(step);
            if (step == 0) step = 1;
            if (step > 100) step = step % 100 + 1; // Limit step size
        }
        
        // Apply slice_copy operation
        try {
            torch::Tensor result = torch::slice_copy(input_tensor, dim, start, end, step);
            
            // Use the result to prevent optimization
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected for invalid parameters
        }
        
        // Try with negative indices (Python-style indexing)
        try {
            int64_t neg_start = -std::abs(start % (dim_size + 1)) - 1;
            int64_t neg_end = -std::abs(end % (dim_size + 1)) - 1;
            
            torch::Tensor result = torch::slice_copy(input_tensor, dim, neg_start, neg_end, step);
            
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // Expected for some invalid parameter combinations
        }
        
        // Try with None-like end (large positive number)
        try {
            torch::Tensor result = torch::slice_copy(input_tensor, dim, 0, std::numeric_limits<int64_t>::max(), step);
            
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // Expected for some cases
        }
        
        // Try slicing from the beginning
        try {
            torch::Tensor result = torch::slice_copy(input_tensor, dim, c10::nullopt, end, step);
            
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // Expected for some cases
        }
        
        // Try slicing to the end
        try {
            torch::Tensor result = torch::slice_copy(input_tensor, dim, start, c10::nullopt, step);
            
            if (result.numel() > 0) {
                volatile float sum = result.sum().item<float>();
                (void)sum;
            }
        } catch (const c10::Error& e) {
            // Expected for some cases
        }
        
        // Try with different dimensions
        for (int64_t d = 0; d < input_tensor.dim() && d < 4; d++) {
            try {
                torch::Tensor result = torch::slice_copy(input_tensor, d, 0, input_tensor.size(d), 1);
                
                if (result.numel() > 0) {
                    volatile float sum = result.sum().item<float>();
                    (void)sum;
                }
            } catch (const c10::Error& e) {
                // Expected for some cases
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