#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Method 1: Create a Long tensor and access its storage
        torch::Tensor long_tensor = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kLong);
        
        // Access storage from tensor
        c10::Storage storage = long_tensor.storage();
        
        // Test storage properties
        size_t storage_size = storage.nbytes();
        
        // Test data pointer access
        if (storage_size > 0) {
            int64_t* data_ptr = static_cast<int64_t*>(storage.mutable_data());
            if (data_ptr != nullptr && long_tensor.numel() > 0) {
                // Read first element
                int64_t first_val = data_ptr[0];
                (void)first_val;
            }
        }
        
        // Method 2: Create storage with specific size
        if (offset < Size) {
            uint8_t num_elements = Data[offset++] % 32 + 1; // 1-32 elements
            
            try {
                // Create a Long tensor of specific size
                torch::Tensor sized_tensor = torch::zeros({num_elements}, torch::kLong);
                c10::Storage sized_storage = sized_tensor.storage();
                
                // Fill storage with fuzzed values
                int64_t* ptr = static_cast<int64_t*>(sized_storage.mutable_data());
                for (int i = 0; i < num_elements && offset + sizeof(int64_t) <= Size; i++) {
                    int64_t value;
                    std::memcpy(&value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    ptr[i] = value;
                }
                
                // Test tensor after storage modification
                torch::Tensor result = sized_tensor + 1;
                (void)result;
            }
            catch (const c10::Error &e) {
                // Expected for some operations
            }
        }
        
        // Method 3: Test storage sharing between tensors
        if (offset + 2 < Size) {
            uint8_t dim1 = (Data[offset++] % 10) + 1;
            uint8_t dim2 = (Data[offset++] % 10) + 1;
            
            try {
                torch::Tensor base_tensor = torch::arange(dim1 * dim2, torch::kLong).reshape({dim1, dim2});
                c10::Storage base_storage = base_tensor.storage();
                
                // Create a view that shares storage
                torch::Tensor view_tensor = base_tensor.view({-1});
                c10::Storage view_storage = view_tensor.storage();
                
                // Verify they share the same storage
                bool same_storage = (base_storage.data() == view_storage.data());
                (void)same_storage;
                
                // Modify through view and check base
                if (view_tensor.numel() > 0) {
                    view_tensor[0] = 999;
                    int64_t base_val = base_tensor.flatten()[0].item<int64_t>();
                    (void)base_val;
                }
            }
            catch (const c10::Error &e) {
                // Expected for invalid shapes
            }
        }
        
        // Method 4: Test storage offset
        if (offset + 1 < Size) {
            uint8_t tensor_size = (Data[offset++] % 20) + 2;
            
            try {
                torch::Tensor full_tensor = torch::arange(tensor_size, torch::kLong);
                
                // Create a slice which has storage offset
                torch::Tensor slice = full_tensor.narrow(0, 1, tensor_size - 1);
                
                // Storage offset should be non-zero for slice
                int64_t storage_offset = slice.storage_offset();
                (void)storage_offset;
                
                // Test that slice storage is still the same underlying storage
                c10::Storage full_storage = full_tensor.storage();
                c10::Storage slice_storage = slice.storage();
                bool same_data = (full_storage.data() == slice_storage.data());
                (void)same_data;
            }
            catch (const c10::Error &e) {
                // Expected for invalid operations
            }
        }
        
        // Method 5: Test storage resize through tensor operations
        if (offset + 1 < Size) {
            uint8_t initial_size = (Data[offset++] % 10) + 1;
            
            try {
                torch::Tensor tensor = torch::zeros({initial_size}, torch::kLong);
                size_t original_nbytes = tensor.storage().nbytes();
                
                // Operations that may reallocate storage
                torch::Tensor expanded = tensor.expand({2, initial_size});
                torch::Tensor contiguous = expanded.contiguous();
                
                // New tensor should have its own storage
                size_t new_nbytes = contiguous.storage().nbytes();
                (void)original_nbytes;
                (void)new_nbytes;
            }
            catch (const c10::Error &e) {
                // Expected for invalid operations
            }
        }
        
        // Method 6: Test storage with different memory formats
        if (offset + 2 < Size) {
            uint8_t h = (Data[offset++] % 5) + 1;
            uint8_t w = (Data[offset++] % 5) + 1;
            
            try {
                torch::Tensor nhwc_tensor = torch::zeros({1, h, w, 3}, torch::kLong)
                    .to(torch::MemoryFormat::ChannelsLast);
                
                c10::Storage nhwc_storage = nhwc_tensor.storage();
                size_t storage_bytes = nhwc_storage.nbytes();
                (void)storage_bytes;
                
                // Convert to contiguous and check storage
                torch::Tensor contig = nhwc_tensor.contiguous();
                c10::Storage contig_storage = contig.storage();
                
                // They may or may not share storage depending on layout
                bool shares = (nhwc_storage.data() == contig_storage.data());
                (void)shares;
            }
            catch (const c10::Error &e) {
                // Expected for invalid operations
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