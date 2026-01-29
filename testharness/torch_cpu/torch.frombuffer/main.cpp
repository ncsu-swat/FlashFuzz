#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>
#include <vector>

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
        
        // Need at least a few bytes to create a meaningful buffer
        if (Size < 8) {
            return 0;
        }
        
        // Parse buffer size
        uint32_t buffer_size = 0;
        std::memcpy(&buffer_size, Data + offset, sizeof(buffer_size));
        offset += sizeof(buffer_size);
        
        // Limit buffer size to avoid excessive memory usage (8 to 512 bytes)
        buffer_size = 8 + (buffer_size % 504);
        
        // Create a buffer to use with from_blob
        std::vector<uint8_t> buffer(buffer_size);
        
        // Fill buffer with data from input
        for (uint32_t i = 0; i < buffer_size && offset < Size; i++) {
            buffer[i] = Data[offset++];
        }
        
        // Parse control byte for dtype selection
        uint8_t dtype_byte = 0;
        if (offset < Size) {
            dtype_byte = Data[offset++];
        }
        
        // Parse byte offset within the buffer
        uint8_t offset_byte = 0;
        if (offset < Size) {
            offset_byte = Data[offset++];
        }
        
        // Parse shape control byte
        uint8_t shape_control = 0;
        if (offset < Size) {
            shape_control = Data[offset++];
        }

        // Select dtype based on input - use types that are safe with from_blob
        torch::ScalarType dtype;
        size_t element_size;
        switch (dtype_byte % 6) {
            case 0:
                dtype = torch::kUInt8;
                element_size = 1;
                break;
            case 1:
                dtype = torch::kInt8;
                element_size = 1;
                break;
            case 2:
                dtype = torch::kInt16;
                element_size = 2;
                break;
            case 3:
                dtype = torch::kInt32;
                element_size = 4;
                break;
            case 4:
                dtype = torch::kFloat32;
                element_size = 4;
                break;
            case 5:
                dtype = torch::kFloat64;
                element_size = 8;
                break;
            default:
                dtype = torch::kFloat32;
                element_size = 4;
                break;
        }
        
        // Calculate number of elements that fit in buffer
        size_t num_elements = buffer_size / element_size;
        if (num_elements == 0) {
            return 0;
        }
        
        // Ensure buffer is properly aligned for the dtype
        size_t usable_bytes = num_elements * element_size;
        
        // Test 1: Basic 1D tensor from buffer
        try {
            torch::Tensor tensor1 = torch::from_blob(
                buffer.data(),
                {static_cast<int64_t>(num_elements)},
                torch::TensorOptions().dtype(dtype)
            );
            // Force computation to ensure the tensor is actually used
            auto sum1 = tensor1.sum();
            (void)sum1;
        } catch (const c10::Error& e) {
            // Expected for invalid configurations
        }
        
        // Test 2: 2D tensor with different shapes based on fuzz input
        if (num_elements >= 4) {
            try {
                int64_t dim0 = 2 + (shape_control % 4);  // 2-5
                int64_t dim1 = num_elements / dim0;
                if (dim1 > 0 && dim0 * dim1 <= static_cast<int64_t>(num_elements)) {
                    torch::Tensor tensor2 = torch::from_blob(
                        buffer.data(),
                        {dim0, dim1},
                        torch::TensorOptions().dtype(dtype)
                    );
                    auto sum2 = tensor2.sum();
                    (void)sum2;
                }
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
        }
        
        // Test 3: With byte offset into buffer
        size_t byte_offset = (offset_byte % (buffer_size / 2));
        // Align offset to element size
        byte_offset = (byte_offset / element_size) * element_size;
        size_t remaining_elements = (buffer_size - byte_offset) / element_size;
        
        if (remaining_elements > 0) {
            try {
                torch::Tensor tensor3 = torch::from_blob(
                    buffer.data() + byte_offset,
                    {static_cast<int64_t>(remaining_elements)},
                    torch::TensorOptions().dtype(dtype)
                );
                auto sum3 = tensor3.sum();
                (void)sum3;
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
        }
        
        // Test 4: 3D tensor
        if (num_elements >= 8) {
            try {
                int64_t d0 = 2;
                int64_t d1 = 2;
                int64_t d2 = num_elements / 4;
                if (d2 > 0 && d0 * d1 * d2 <= static_cast<int64_t>(num_elements)) {
                    torch::Tensor tensor4 = torch::from_blob(
                        buffer.data(),
                        {d0, d1, d2},
                        torch::TensorOptions().dtype(dtype)
                    );
                    auto sum4 = tensor4.sum();
                    (void)sum4;
                }
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
        }
        
        // Test 5: With custom strides (non-contiguous view)
        if (num_elements >= 4) {
            try {
                std::vector<int64_t> sizes = {2, static_cast<int64_t>(num_elements / 4)};
                if (sizes[1] > 0) {
                    // Strides that create a strided view
                    std::vector<int64_t> strides = {2, 1};
                    torch::Tensor tensor5 = torch::from_blob(
                        buffer.data(),
                        sizes,
                        strides,
                        torch::TensorOptions().dtype(dtype)
                    );
                    auto sum5 = tensor5.sum();
                    (void)sum5;
                }
            } catch (const c10::Error& e) {
                // Expected for invalid configurations
            }
        }
        
        // Test 6: With a deleter callback
        try {
            auto deleter = [](void* ptr) {
                // No-op deleter since buffer is managed by vector
            };
            torch::Tensor tensor6 = torch::from_blob(
                buffer.data(),
                {static_cast<int64_t>(num_elements)},
                deleter,
                torch::TensorOptions().dtype(dtype)
            );
            auto sum6 = tensor6.sum();
            (void)sum6;
        } catch (const c10::Error& e) {
            // Expected for invalid configurations
        }
        
        // Test 7: Clone the tensor to test the data is valid
        try {
            torch::Tensor tensor7 = torch::from_blob(
                buffer.data(),
                {static_cast<int64_t>(num_elements)},
                torch::TensorOptions().dtype(dtype)
            );
            // Clone creates a copy, ensuring data is read
            torch::Tensor cloned = tensor7.clone();
            auto sum7 = cloned.sum();
            (void)sum7;
        } catch (const c10::Error& e) {
            // Expected for invalid configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}