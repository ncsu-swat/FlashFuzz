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
            // Keep dimensions reasonable to avoid memory issues
            int64_t dim_size = (Data[offset++] % 100) + 1;
            sizes.push_back(dim_size);
        }

        if (offset >= Size) {
            return 0;
        }

        // Extract dtype choice
        uint8_t dtype_choice = Data[offset++] % 4;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kBFloat16; break;
        }

        // Extract device choice
        torch::Device device = torch::kCPU;
        if (offset < Size) {
            uint8_t device_choice = Data[offset++] % 2;
            if (device_choice == 1 && torch::cuda::is_available()) {
                device = torch::kCUDA;
            }
        }

        // Extract layout choice
        torch::Layout layout = torch::kStrided;
        if (offset < Size) {
            uint8_t layout_choice = Data[offset++] % 2;
            if (layout_choice == 1) {
                layout = torch::kSparse;
            }
        }

        // Extract requires_grad
        bool requires_grad = false;
        if (offset < Size) {
            requires_grad = (Data[offset++] % 2) == 1;
        }

        // Test different torch::randn overloads
        
        // 1. Basic randn with IntArrayRef
        auto tensor1 = torch::randn(sizes);
        
        // 2. randn with TensorOptions
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .layout(layout)
            .requires_grad(requires_grad);
        
        auto tensor2 = torch::randn(sizes, options);
        
        // 3. randn with individual options
        auto tensor3 = torch::randn(sizes, dtype);
        
        // 4. Test with generator if we have enough data
        torch::Generator gen;
        if (offset + 8 <= Size) {
            uint64_t seed = 0;
            for (int i = 0; i < 8 && offset < Size; i++) {
                seed = (seed << 8) | Data[offset++];
            }
            gen = torch::default_generator();
            gen.set_current_seed(seed);
            
            auto tensor4 = torch::randn(sizes, gen);
            auto tensor5 = torch::randn(sizes, gen, options);
        }

        // Test edge cases
        
        // Empty tensor
        auto empty_tensor = torch::randn({0});
        
        // Single element tensor
        auto single_tensor = torch::randn({1});
        
        // Large dimension count (but small sizes)
        std::vector<int64_t> many_dims(6, 1);
        auto many_dims_tensor = torch::randn(many_dims);
        
        // Test operations on generated tensors to ensure they're valid
        if (tensor1.numel() > 0) {
            auto mean_val = tensor1.mean();
            auto std_val = tensor1.std();
            auto sum_val = tensor1.sum();
        }
        
        if (tensor2.numel() > 0) {
            auto reshaped = tensor2.reshape({-1});
            auto cloned = tensor2.clone();
        }

        // Test different size specifications
        if (offset < Size) {
            // Test with different size patterns
            uint8_t size_pattern = Data[offset++] % 4;
            switch (size_pattern) {
                case 0: {
                    // Square matrix
                    int64_t n = (offset < Size) ? (Data[offset++] % 50) + 1 : 10;
                    auto square = torch::randn({n, n});
                    break;
                }
                case 1: {
                    // 3D tensor
                    if (offset + 2 < Size) {
                        int64_t d1 = (Data[offset++] % 20) + 1;
                        int64_t d2 = (Data[offset++] % 20) + 1;
                        int64_t d3 = (Data[offset++] % 20) + 1;
                        auto tensor_3d = torch::randn({d1, d2, d3});
                    }
                    break;
                }
                case 2: {
                    // Vector
                    int64_t len = (offset < Size) ? (Data[offset++] % 1000) + 1 : 100;
                    auto vector = torch::randn({len});
                    break;
                }
                default: {
                    // Batch of matrices
                    if (offset + 2 < Size) {
                        int64_t batch = (Data[offset++] % 10) + 1;
                        int64_t rows = (Data[offset++] % 50) + 1;
                        int64_t cols = (Data[offset++] % 50) + 1;
                        auto batch_tensor = torch::randn({batch, rows, cols});
                    }
                    break;
                }
            }
        }

        // Test with different memory formats if available
        if (offset < Size && sizes.size() == 4) {
            uint8_t memory_format_choice = Data[offset++] % 3;
            torch::MemoryFormat memory_format;
            switch (memory_format_choice) {
                case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                case 1: memory_format = torch::MemoryFormat::ChannelsLast; break;
                default: memory_format = torch::MemoryFormat::Preserve; break;
            }
            
            auto options_with_memory = torch::TensorOptions()
                .dtype(dtype)
                .device(device)
                .memory_format(memory_format);
            
            auto tensor_with_memory = torch::randn(sizes, options_with_memory);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}