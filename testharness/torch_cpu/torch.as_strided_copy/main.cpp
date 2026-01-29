#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>        // For uint64_t

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
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor - make it contiguous for as_strided_copy
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input tensor is valid and contiguous
        if (!input_tensor.defined() || input_tensor.numel() == 0) {
            return 0;
        }
        input_tensor = input_tensor.contiguous();
        
        // Parse parameters for as_strided_copy
        // Need at least 1 byte for rank of size
        if (offset + 1 >= Size) {
            return 0;
        }
        
        // Parse rank for size (limit to reasonable values)
        uint8_t size_rank_byte = Data[offset++];
        uint8_t size_rank = fuzzer_utils::parseRank(size_rank_byte);
        if (size_rank == 0) {
            size_rank = 1; // At least 1D
        }
        if (size_rank > 4) {
            size_rank = 4; // Limit dimensions
        }
        
        // Parse size vector
        std::vector<int64_t> size = fuzzer_utils::parseShape(Data, offset, Size, size_rank);
        
        // Ensure all size values are positive and reasonable
        int64_t total_elements = 1;
        for (auto& s : size) {
            if (s <= 0) s = 1;
            if (s > 64) s = 64; // Limit dimension sizes
            total_elements *= s;
        }
        
        // Parse stride vector - use same rank as size
        std::vector<int64_t> stride;
        stride.reserve(size_rank);
        
        for (size_t i = 0; i < size_rank; i++) {
            int64_t s = 1;
            if (offset + 1 <= Size) {
                s = static_cast<int64_t>(Data[offset++] % 16) + 1; // Stride between 1-16
            }
            stride.push_back(s);
        }
        
        // Parse storage_offset (keep it reasonable)
        int64_t storage_offset = 0;
        if (offset + 1 <= Size) {
            storage_offset = static_cast<int64_t>(Data[offset++] % 32); // 0-31
        }
        
        // Calculate maximum index that would be accessed
        int64_t max_index = storage_offset;
        for (size_t i = 0; i < size.size(); i++) {
            if (size[i] > 0 && stride[i] > 0) {
                max_index += (size[i] - 1) * stride[i];
            }
        }
        
        // Check if the view would be valid (within storage bounds)
        int64_t storage_size = input_tensor.numel();
        if (max_index >= storage_size) {
            // Adjust to make it valid - scale down or use smaller view
            if (storage_size <= 0) {
                return 0;
            }
            // Create a simple valid view
            size = {std::min((int64_t)4, storage_size)};
            stride = {1};
            storage_offset = 0;
        }
        
        // Apply as_strided_copy operation
        try {
            torch::Tensor result = torch::as_strided_copy(input_tensor, size, stride, storage_offset);
            
            // Perform some operations on the result to ensure it's used
            if (result.defined() && result.numel() > 0) {
                auto sum = result.sum();
                auto mean = result.mean();
                // std::dev may fail on small tensors, wrap separately
                try {
                    auto std_dev = result.std();
                } catch (...) {
                    // std may fail on single element tensors
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and handled
        }
        
        // Also test without storage_offset
        try {
            torch::Tensor result2 = torch::as_strided_copy(input_tensor, size, stride);
            if (result2.defined()) {
                auto sum = result2.sum();
            }
        } catch (const c10::Error& e) {
            // Expected for invalid combinations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}