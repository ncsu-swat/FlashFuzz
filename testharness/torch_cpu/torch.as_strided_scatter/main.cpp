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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create source tensor (to be scattered into input)
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we've consumed all data, create a simple source tensor
            src = torch::ones_like(input);
        }
        
        // Parse parameters for as_strided_scatter
        // Get size for new shape
        std::vector<int64_t> size;
        uint8_t num_dims = 0;
        if (offset < Size) {
            num_dims = Data[offset++] % 5; // 0-4 dimensions
            
            for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; i++) {
                int64_t dim_size;
                std::memcpy(&dim_size, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Bound the dimension size to avoid excessive memory usage
                dim_size = std::abs(dim_size) % 16;
                size.push_back(dim_size);
            }
        }
        
        // If no valid size was parsed, use a default
        if (size.empty() && num_dims > 0) {
            size.push_back(2);
        }
        
        // Get stride for the view
        std::vector<int64_t> stride;
        if (offset < Size) {
            for (uint8_t i = 0; i < num_dims && offset + sizeof(int64_t) <= Size; i++) {
                int64_t stride_val;
                std::memcpy(&stride_val, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow negative strides to test that case
                stride.push_back(stride_val);
            }
        }
        
        // If no valid stride was parsed, use a default
        if (stride.empty() && num_dims > 0) {
            stride.push_back(1);
        }
        
        // Get storage offset
        int64_t storage_offset = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_offset, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try the operation with various parameters
        try {
            // Case 1: Basic usage
            torch::Tensor result = torch::as_strided_scatter(input, src, size, stride, storage_offset);
            
            // Prevent result from being optimized away
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations are fine
        }
        
        // Case 2: Try with empty size/stride
        if (!input.sizes().empty()) {
            try {
                std::vector<int64_t> empty_size;
                std::vector<int64_t> empty_stride;
                torch::Tensor result = torch::as_strided_scatter(input, src, empty_size, empty_stride, storage_offset);
                
                if (result.numel() > 0) {
                    volatile float dummy = result.sum().item<float>();
                    (void)dummy;
                }
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Case 3: Try with mismatched size/stride lengths
        if (size.size() > 1) {
            try {
                std::vector<int64_t> shorter_stride(stride.begin(), stride.begin() + stride.size()/2);
                torch::Tensor result = torch::as_strided_scatter(input, src, size, shorter_stride, storage_offset);
                
                if (result.numel() > 0) {
                    volatile float dummy = result.sum().item<float>();
                    (void)dummy;
                }
            } catch (const c10::Error& e) {
                // Expected exceptions from PyTorch operations are fine
            }
        }
        
        // Case 4: Try with negative storage offset
        try {
            torch::Tensor result = torch::as_strided_scatter(input, src, size, stride, -storage_offset);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations are fine
        }
        
        // Case 5: Try with large storage offset
        try {
            int64_t large_offset = std::numeric_limits<int16_t>::max();
            torch::Tensor result = torch::as_strided_scatter(input, src, size, stride, large_offset);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch operations are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
