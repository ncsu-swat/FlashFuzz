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
        
        // Extract number of dimensions (1-6 to avoid excessive memory usage)
        uint8_t num_dims = (Data[offset++] % 6) + 1;
        
        // Extract dimensions
        std::vector<int64_t> sizes;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Limit dimension size to avoid OOM (max 100 per dimension)
            int64_t dim_size = (Data[offset++] % 100) + 1;
            sizes.push_back(dim_size);
        }
        
        if (offset >= Size) {
            return 0;
        }
        
        // Extract dtype choice
        uint8_t dtype_choice = Data[offset++] % 12;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kUInt8; break;
            case 6: dtype = torch::kInt16; break;
            case 7: dtype = torch::kBool; break;
            case 8: dtype = torch::kComplexFloat; break;
            case 9: dtype = torch::kComplexDouble; break;
            case 10: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }
        
        if (offset >= Size) {
            return 0;
        }
        
        // Extract device choice
        uint8_t device_choice = Data[offset++] % 2;
        torch::Device device = (device_choice == 0) ? torch::kCPU : torch::kCUDA;
        
        if (offset >= Size) {
            return 0;
        }
        
        // Extract requires_grad
        bool requires_grad = (Data[offset++] % 2) == 1;
        
        if (offset >= Size) {
            return 0;
        }
        
        // Extract pin_memory (only valid for CPU)
        bool pin_memory = (Data[offset++] % 2) == 1 && device.is_cpu();
        
        if (offset >= Size) {
            return 0;
        }
        
        // Extract memory format choice
        uint8_t memory_format_choice = Data[offset++] % 4;
        torch::MemoryFormat memory_format;
        switch (memory_format_choice) {
            case 0: memory_format = torch::MemoryFormat::Contiguous; break;
            case 1: memory_format = torch::MemoryFormat::Preserve; break;
            case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
            default: memory_format = torch::MemoryFormat::Contiguous; break;
        }
        
        // Test basic torch::empty with sizes vector
        auto options = torch::TensorOptions()
                          .dtype(dtype)
                          .device(device)
                          .requires_grad(requires_grad)
                          .pinned_memory(pin_memory)
                          .memory_format(memory_format);
        
        // Skip CUDA operations if CUDA is not available
        if (device.is_cuda() && !torch::cuda::is_available()) {
            options = options.device(torch::kCPU);
        }
        
        // Test with vector of sizes
        torch::Tensor tensor1 = torch::empty(sizes, options);
        
        // Verify tensor properties
        if (tensor1.dim() != static_cast<int64_t>(num_dims)) {
            throw std::runtime_error("Dimension mismatch");
        }
        
        for (size_t i = 0; i < sizes.size(); i++) {
            if (tensor1.size(i) != sizes[i]) {
                throw std::runtime_error("Size mismatch");
            }
        }
        
        // Test with IntArrayRef
        torch::IntArrayRef size_ref(sizes);
        torch::Tensor tensor2 = torch::empty(size_ref, options);
        
        // Test edge cases
        if (num_dims == 1 && sizes[0] <= 10) {
            // Test with individual size arguments for small tensors
            switch (num_dims) {
                case 1:
                    torch::empty({sizes[0]}, options);
                    break;
            }
        }
        
        // Test with different combinations
        if (sizes.size() >= 2) {
            // Test 2D case
            torch::empty({sizes[0], sizes[1]}, options);
        }
        
        if (sizes.size() >= 3) {
            // Test 3D case  
            torch::empty({sizes[0], sizes[1], sizes[2]}, options);
        }
        
        // Test zero-sized tensors
        if (offset < Size && (Data[offset++] % 10) == 0) {
            torch::empty({0}, options);
            torch::empty({0, 5}, options);
            torch::empty({5, 0}, options);
        }
        
        // Test very small tensors
        torch::empty({1}, options);
        torch::empty({1, 1}, options);
        
        // Test with default options
        torch::empty(sizes);
        
        // Test memory format compatibility
        if (sizes.size() == 4 && memory_format == torch::MemoryFormat::ChannelsLast) {
            // ChannelsLast is only valid for 4D tensors
            torch::empty(sizes, options);
        }
        
        // Access tensor data to ensure it's properly allocated
        if (tensor1.numel() > 0 && tensor1.numel() < 1000) {
            // Only access small tensors to avoid performance issues
            tensor1.data_ptr();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}