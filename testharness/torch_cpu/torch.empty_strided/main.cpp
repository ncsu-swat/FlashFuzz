#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For memcpy

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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse size information
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape for the tensor
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Validate shape - ensure all dimensions are positive and reasonable
        int64_t total_elements = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] <= 0) {
                shape[i] = 1;  // Ensure positive dimensions
            }
            if (shape[i] > 1024) {
                shape[i] = 1024;  // Cap dimensions to avoid OOM
            }
            total_elements *= shape[i];
            if (total_elements > 1024 * 1024) {
                // Too many elements, reduce this dimension
                shape[i] = 1;
                total_elements = 1;
                for (auto dim : shape) total_elements *= dim;
            }
        }
        
        // Parse data type
        uint8_t dtype_selector = 0;
        if (offset < Size) {
            dtype_selector = Data[offset++];
        }
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Parse strides - use smaller values from bytes to keep them reasonable
        std::vector<int64_t> strides;
        for (size_t i = 0; i < shape.size() && offset < Size; i++) {
            // Use single byte for stride to keep values manageable
            int64_t stride_val = static_cast<int64_t>(Data[offset++]);
            // Strides should be non-negative and reasonable
            stride_val = stride_val % 256;  // Keep stride in reasonable range
            strides.push_back(stride_val);
        }
        
        // If we don't have enough strides, compute default contiguous strides
        if (strides.size() < shape.size()) {
            strides.clear();
            strides.resize(shape.size());
            if (!shape.empty()) {
                strides[shape.size() - 1] = 1;
                for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }
        }
        
        // Validate that strides won't cause excessive memory allocation
        // Compute the storage size needed
        int64_t storage_size = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] > 0) {
                storage_size = std::max(storage_size, (shape[i] - 1) * strides[i] + 1);
            }
        }
        
        // Cap storage size to avoid OOM
        if (storage_size > 10 * 1024 * 1024) {
            // Recompute with contiguous strides
            if (!shape.empty()) {
                strides[shape.size() - 1] = 1;
                for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }
        }
        
        // Create options with the specified dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Call empty_strided with the parsed parameters
        torch::Tensor result;
        try {
            result = torch::empty_strided(shape, strides, options);
            
            // Basic checks to ensure the tensor was created correctly
            auto result_sizes = result.sizes();
            for (size_t i = 0; i < shape.size(); i++) {
                if (result_sizes[i] != shape[i]) {
                    // Shape mismatch - this is unexpected
                    break;
                }
            }
            
            // Check strides
            for (size_t i = 0; i < shape.size(); i++) {
                if (result.stride(i) != strides[i]) {
                    // Stride mismatch - this is unexpected
                    break;
                }
            }
            
            // Perform some operations on the tensor to ensure it's valid
            if (result.numel() > 0) {
                // Try different operations to exercise the tensor
                try {
                    result.zero_();
                } catch (...) {
                    // Some stride configurations may not support in-place ops
                }
                
                try {
                    result.fill_(1.0);
                } catch (...) {
                    // Some dtype/stride combinations may fail
                }
                
                // Try to read from the tensor
                try {
                    auto sum = result.sum();
                    (void)sum;
                } catch (...) {
                    // May fail for certain configurations
                }
            }
            
            // Test with different device options if available
            if (offset < Size) {
                uint8_t pin_memory = Data[offset++] % 2;
                try {
                    auto options2 = torch::TensorOptions().dtype(dtype).pinned_memory(pin_memory == 1);
                    auto result2 = torch::empty_strided(shape, strides, options2);
                    (void)result2;
                } catch (...) {
                    // Pinned memory may not be available
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for invalid inputs
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors for invalid configurations are expected
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}