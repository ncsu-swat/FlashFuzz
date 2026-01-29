#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstdint>

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - ensure it has some storage to work with
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the storage size of the input tensor
        int64_t storage_size = input_tensor.storage().nbytes() / input_tensor.element_size();
        if (storage_size <= 0) {
            return 0;
        }
        
        // Parse number of dimensions for the new size (limit to reasonable range)
        uint8_t size_rank = 1;
        if (offset < Size) {
            size_rank = (Data[offset++] % 4) + 1; // 1-4 dimensions
        }
        
        // Parse the new size
        std::vector<int64_t> new_size;
        for (uint8_t i = 0; i < size_rank && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 8) + 1; // 1-8 per dimension
            new_size.push_back(dim_size);
        }
        
        // Ensure we have at least one dimension
        if (new_size.empty()) {
            new_size.push_back(1);
        }
        
        // Parse the new stride - same number of dimensions as size
        std::vector<int64_t> new_stride;
        for (size_t i = 0; i < new_size.size() && offset < Size; i++) {
            int64_t stride_val = (Data[offset++] % 4) + 1; // 1-4 stride values
            new_stride.push_back(stride_val);
        }
        
        // If stride is incomplete, fill with default contiguous strides
        while (new_stride.size() < new_size.size()) {
            if (new_stride.empty()) {
                new_stride.push_back(1);
            } else {
                // Insert at the beginning for contiguous layout
                new_stride.insert(new_stride.begin(), new_stride[0] * new_size[new_stride.size() - 1]);
            }
        }
        
        // Compute contiguous strides from scratch to ensure validity
        new_stride.clear();
        new_stride.resize(new_size.size());
        if (!new_size.empty()) {
            new_stride[new_size.size() - 1] = 1;
            for (int i = static_cast<int>(new_size.size()) - 2; i >= 0; i--) {
                new_stride[i] = new_stride[i + 1] * new_size[i + 1];
            }
        }
        
        // Parse storage offset - constrain to valid range
        int64_t storage_offset = 0;
        if (offset < Size) {
            storage_offset = Data[offset++] % std::max<int64_t>(1, storage_size / 2);
        }
        
        // Calculate the maximum index that will be accessed
        int64_t max_index = storage_offset;
        for (size_t i = 0; i < new_size.size(); i++) {
            max_index += (new_size[i] - 1) * new_stride[i];
        }
        
        // Only proceed if the view fits within the storage
        if (max_index >= storage_size) {
            // Adjust size to fit
            for (size_t i = 0; i < new_size.size(); i++) {
                new_size[i] = 1;
            }
            storage_offset = 0;
        }
        
        // Clone the tensor to work with
        torch::Tensor tensor_copy = input_tensor.clone();
        
        // Apply as_strided_ operation (in-place)
        try {
            tensor_copy.as_strided_(new_size, new_stride, storage_offset);
            
            // Access the tensor to ensure the view is valid
            if (tensor_copy.numel() > 0) {
                // Sum all elements instead of item() which requires exactly 1 element
                auto sum_result = tensor_copy.sum();
                (void)sum_result;
            }
            
            // Try some operations on the strided view
            if (tensor_copy.numel() > 0 && tensor_copy.numel() <= 64) {
                auto contiguous = tensor_copy.contiguous();
                (void)contiguous;
            }
        }
        catch (const c10::Error &e) {
            // Expected failures due to invalid strides/sizes - silently ignore
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