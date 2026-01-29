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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip if input is too small or has no dimensions
        if (input.numel() == 0 || input.dim() == 0) {
            return 0;
        }
        
        // Get a dimension to scatter along (must be valid)
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % input.dim();
        }
        
        // Get the size along the scatter dimension
        int64_t dim_size = input.size(dim);
        if (dim_size == 0) {
            return 0;
        }
        
        // Create index tensor with same shape as input but with valid indices
        // Index values must be in range [0, dim_size) for the scatter dimension
        torch::Tensor index;
        if (offset < Size) {
            index = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure index has same number of dimensions as input
            while (index.dim() < input.dim()) {
                index = index.unsqueeze(0);
            }
            while (index.dim() > input.dim()) {
                index = index.squeeze(0);
                if (index.dim() == 0) {
                    index = index.unsqueeze(0);
                    break;
                }
            }
            // Convert to long and clamp to valid range
            index = index.to(torch::kLong).abs();
            index = index.remainder(dim_size);
        } else {
            index = torch::zeros_like(input, torch::kLong);
        }
        
        // Create src tensor - must broadcast to index shape
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
            // Try to make src compatible with index shape
            src = src.to(input.dtype());
        } else {
            src = torch::ones_like(input);
        }
        
        // Get operation selector
        uint8_t op_selector = 0;
        if (offset < Size) {
            op_selector = Data[offset++];
        }
        
        // Try scatter (out-of-place)
        try {
            torch::Tensor result = input.scatter(dim, index, src);
        } catch (const std::exception&) {
            // Expected for some invalid input combinations
        }
        
        // Try scatter with scalar value
        try {
            double value = 1.0;
            if (offset + sizeof(float) <= Size) {
                float fval;
                std::memcpy(&fval, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Avoid NaN/Inf
                if (std::isfinite(fval)) {
                    value = static_cast<double>(fval);
                }
            }
            torch::Tensor result = input.scatter(dim, index, value);
        } catch (const std::exception&) {
            // Expected for some invalid input combinations
        }
        
        // Try scatter_ (in-place)
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.scatter_(dim, index, src);
        } catch (const std::exception&) {
            // Expected for some invalid input combinations
        }
        
        // Try scatter_ with scalar value (in-place)
        try {
            double value = 2.0;
            if (offset + sizeof(float) <= Size) {
                float fval;
                std::memcpy(&fval, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isfinite(fval)) {
                    value = static_cast<double>(fval);
                }
            }
            torch::Tensor input_copy = input.clone();
            input_copy.scatter_(dim, index, value);
        } catch (const std::exception&) {
            // Expected for some invalid input combinations
        }
        
        // Try different reduction modes
        if (offset < Size) {
            uint8_t reduce_selector = Data[offset++] % 3;
            
            // Test "add" reduction
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.scatter_(dim, index, src, "add");
            } catch (const std::exception&) {
                // Expected for some invalid input combinations
            }
            
            // Test "multiply" reduction  
            try {
                torch::Tensor input_copy = input.clone();
                input_copy.scatter_(dim, index, src, "multiply");
            } catch (const std::exception&) {
                // Expected for some invalid input combinations
            }
        }
        
        // Also test torch::scatter function (not just method)
        try {
            torch::Tensor result = torch::scatter(input, dim, index, src);
        } catch (const std::exception&) {
            // Expected for some invalid input combinations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}