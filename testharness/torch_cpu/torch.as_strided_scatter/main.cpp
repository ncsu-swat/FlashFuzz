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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor - needs to be large enough to support strided views
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has enough elements for interesting strided operations
        if (input.numel() < 1) {
            return 0;
        }
        
        // Parse parameters for as_strided_scatter
        // Get size for new shape (at least 1 dimension)
        std::vector<int64_t> size;
        uint8_t num_dims = 1;
        if (offset < Size) {
            num_dims = 1 + (Data[offset++] % 4); // 1-4 dimensions
        }
        
        int64_t total_elements = 1;
        for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
            int64_t dim_size = 1 + (Data[offset++] % 8); // 1-8 per dimension
            size.push_back(dim_size);
            total_elements *= dim_size;
        }
        
        // Pad with 1s if we didn't get enough dimension sizes
        while (size.size() < num_dims) {
            size.push_back(1);
        }
        
        // Get stride for the view
        std::vector<int64_t> stride;
        for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
            int64_t stride_val = 1 + (Data[offset++] % 4); // 1-4 stride
            stride.push_back(stride_val);
        }
        
        // Pad with 1s if we didn't get enough strides
        while (stride.size() < num_dims) {
            stride.push_back(1);
        }
        
        // Get storage offset (bounded to avoid out-of-bounds)
        int64_t storage_offset = 0;
        if (offset < Size) {
            storage_offset = Data[offset++] % std::max<int64_t>(1, input.numel() / 2);
        }
        
        // Create source tensor that matches the strided view shape
        torch::Tensor src = torch::randn(size, input.options());
        
        // Case 1: Basic usage with matching src shape
        try {
            torch::Tensor result = torch::as_strided_scatter(input, src, size, stride, storage_offset);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions from invalid parameters
        }
        
        // Case 2: Try with zero storage offset
        try {
            torch::Tensor result = torch::as_strided_scatter(input, src, size, stride, 0);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Case 3: Try with contiguous strides
        try {
            std::vector<int64_t> contig_stride;
            int64_t s = 1;
            for (int i = num_dims - 1; i >= 0; i--) {
                contig_stride.insert(contig_stride.begin(), s);
                s *= size[i];
            }
            torch::Tensor result = torch::as_strided_scatter(input, src, size, contig_stride, storage_offset);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Case 4: Try scalar case (0-d tensor)
        try {
            std::vector<int64_t> scalar_size;
            std::vector<int64_t> scalar_stride;
            torch::Tensor scalar_src = torch::randn({}, input.options());
            torch::Tensor result = torch::as_strided_scatter(input, scalar_src, scalar_size, scalar_stride, storage_offset);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Case 5: Try 1-d case
        try {
            int64_t len = 1 + (offset < Size ? Data[offset] % 8 : 2);
            std::vector<int64_t> size_1d = {len};
            std::vector<int64_t> stride_1d = {1};
            torch::Tensor src_1d = torch::randn(size_1d, input.options());
            torch::Tensor result = torch::as_strided_scatter(input, src_1d, size_1d, stride_1d, 0);
            
            if (result.numel() > 0) {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
        
        // Case 6: Try with different dtypes
        try {
            torch::Tensor input_int = input.to(torch::kInt32);
            torch::Tensor src_int = src.to(torch::kInt32);
            torch::Tensor result = torch::as_strided_scatter(input_int, src_int, size, stride, storage_offset);
            
            if (result.numel() > 0) {
                volatile int dummy = result.sum().item<int>();
                (void)dummy;
            }
        } catch (const c10::Error& e) {
            // Expected exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}