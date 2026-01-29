#include "fuzzer_utils.h"
#include <iostream>
#include <cstdint>

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor with some minimum size to allow interesting strided views
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is contiguous and has storage
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Parse parameters for as_strided
        if (Size - offset < 2) {
            return 0;
        }
        
        // Parse size (shape) for as_strided
        uint8_t size_rank = (Data[offset++] % 4) + 1; // Limit rank to 1-4 (avoid empty)
        std::vector<int64_t> size_vec = fuzzer_utils::parseShape(Data, offset, Size, size_rank);
        
        // Ensure all dimensions are positive
        for (auto& dim : size_vec) {
            if (dim <= 0) dim = 1;
            if (dim > 16) dim = 16; // Limit dimension sizes
        }
        
        // Parse strides for as_strided - must match size dimensions
        std::vector<int64_t> stride_vec;
        for (size_t i = 0; i < size_vec.size(); i++) {
            int64_t stride = 1;
            if (offset + 1 <= Size) {
                stride = static_cast<int64_t>(Data[offset++] % 8) + 1; // Positive strides 1-8
            }
            stride_vec.push_back(stride);
        }
        
        // Calculate required storage size for the view
        int64_t required_storage = 0;
        for (size_t i = 0; i < size_vec.size(); i++) {
            required_storage += (size_vec[i] - 1) * stride_vec[i];
        }
        required_storage += 1; // For the element at index 0
        
        // Parse storage_offset - must be valid
        int64_t storage_offset = 0;
        if (offset + 1 <= Size) {
            storage_offset = Data[offset++] % 4; // Small offset 0-3
        }
        
        // Validate that the view fits within storage
        int64_t total_required = storage_offset + required_storage;
        if (total_required > input_tensor.numel()) {
            // Skip invalid combinations that would access out-of-bounds
            return 0;
        }

        // Inner try-catch for expected as_strided failures
        try {
            // Apply as_strided operation
            torch::Tensor result = input_tensor.as_strided(size_vec, stride_vec, storage_offset);
            
            // Basic validation - compute sum to exercise the strided view
            if (result.defined() && result.numel() > 0) {
                auto sum = result.sum();
                
                // Try cloning to force data access through strides
                auto cloned = result.clone();
                
                // Test arithmetic on strided tensor
                auto doubled = result * 2;
                
                // Test contiguous conversion
                auto contig = result.contiguous();
            }
        }
        catch (const c10::Error&) {
            // Expected failures from invalid stride/size combinations
            return 0;
        }
        catch (const std::runtime_error&) {
            // Expected runtime errors
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