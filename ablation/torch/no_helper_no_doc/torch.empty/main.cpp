#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some bytes for basic fuzzing
        if (Size < 8) {
            return 0;
        }
        
        // Extract number of dimensions (1-6 dimensions to keep it reasonable)
        uint8_t num_dims = (Data[offset++] % 6) + 1;
        
        // Extract dimensions
        std::vector<int64_t> sizes;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Use 2 bytes per dimension, limit size to reasonable range
            uint16_t dim_size = 0;
            if (offset + 1 < Size) {
                dim_size = (Data[offset] | (Data[offset + 1] << 8)) % 1000 + 1; // 1-1000
                offset += 2;
            } else {
                dim_size = 1;
            }
            sizes.push_back(static_cast<int64_t>(dim_size));
        }
        
        // Extract dtype (limit to common types)
        torch::ScalarType dtype = torch::kFloat32; // default
        if (offset < Size) {
            uint8_t dtype_idx = Data[offset++] % 8;
            switch (dtype_idx) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kInt8; break;
                case 5: dtype = torch::kUInt8; break;
                case 6: dtype = torch::kBool; break;
                case 7: dtype = torch::kFloat16; break;
            }
        }
        
        // Extract device type
        torch::Device device = torch::kCPU; // default
        if (offset < Size) {
            uint8_t device_idx = Data[offset++] % 2;
            if (device_idx == 0) {
                device = torch::kCPU;
            } else {
                // Only try CUDA if available
                if (torch::cuda::is_available()) {
                    device = torch::kCUDA;
                }
            }
        }
        
        // Extract memory format
        torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous; // default
        if (offset < Size) {
            uint8_t format_idx = Data[offset++] % 3;
            switch (format_idx) {
                case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                case 1: memory_format = torch::MemoryFormat::Preserve; break;
                case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
            }
        }
        
        // Extract requires_grad flag
        bool requires_grad = false;
        if (offset < Size) {
            requires_grad = (Data[offset++] % 2) == 1;
        }
        
        // Test torch::empty with various parameter combinations
        
        // Test 1: Basic empty tensor with sizes only
        auto tensor1 = torch::empty(sizes);
        
        // Test 2: Empty tensor with dtype
        auto tensor2 = torch::empty(sizes, torch::TensorOptions().dtype(dtype));
        
        // Test 3: Empty tensor with device
        auto tensor3 = torch::empty(sizes, torch::TensorOptions().device(device));
        
        // Test 4: Empty tensor with dtype and device
        auto tensor4 = torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
        
        // Test 5: Empty tensor with requires_grad
        auto tensor5 = torch::empty(sizes, torch::TensorOptions().requires_grad(requires_grad));
        
        // Test 6: Empty tensor with memory format (if supported)
        try {
            auto tensor6 = torch::empty(sizes, torch::TensorOptions().memory_format(memory_format));
        } catch (...) {
            // Some memory formats might not be supported for all tensor shapes
        }
        
        // Test 7: Empty tensor with all options
        try {
            auto tensor7 = torch::empty(sizes, torch::TensorOptions()
                                                .dtype(dtype)
                                                .device(device)
                                                .requires_grad(requires_grad)
                                                .memory_format(memory_format));
        } catch (...) {
            // Some combinations might not be valid
        }
        
        // Test edge cases
        
        // Test 8: Empty tensor with zero-sized dimensions
        if (offset < Size && (Data[offset++] % 4) == 0) {
            std::vector<int64_t> zero_sizes = sizes;
            if (!zero_sizes.empty()) {
                zero_sizes[0] = 0; // Make first dimension zero
                auto tensor8 = torch::empty(zero_sizes);
            }
        }
        
        // Test 9: Single element tensor
        if (offset < Size && (Data[offset++] % 4) == 1) {
            std::vector<int64_t> single_sizes(num_dims, 1);
            auto tensor9 = torch::empty(single_sizes, torch::TensorOptions().dtype(dtype));
        }
        
        // Test 10: Large dimension count with small sizes
        if (offset < Size && (Data[offset++] % 4) == 2) {
            std::vector<int64_t> many_small_dims(6, 2);
            auto tensor10 = torch::empty(many_small_dims);
        }
        
        // Verify basic properties of created tensors
        if (tensor1.defined()) {
            auto tensor1_sizes = tensor1.sizes();
            auto tensor1_dtype = tensor1.scalar_type();
            auto tensor1_device = tensor1.device();
        }
        
        // Test IntArrayRef version with C-style array
        if (sizes.size() <= 4) { // Limit for stack allocation
            int64_t c_sizes[4];
            for (size_t i = 0; i < sizes.size(); i++) {
                c_sizes[i] = sizes[i];
            }
            auto tensor_c = torch::empty(torch::IntArrayRef(c_sizes, sizes.size()));
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}