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
        if (Size < 16) {
            return 0;
        }

        // Create input tensor with fuzzed properties
        auto input_tensor = create_random_tensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        // Test basic empty_like
        auto result1 = torch::empty_like(input_tensor);
        
        // Verify basic properties
        if (!result1.sizes().equals(input_tensor.sizes())) {
            std::cerr << "Size mismatch in basic empty_like" << std::endl;
        }

        // Test with different dtype if we have enough data
        if (offset + 1 < Size) {
            auto dtype_choice = Data[offset++] % 8;
            torch::ScalarType target_dtype;
            
            switch (dtype_choice) {
                case 0: target_dtype = torch::kFloat32; break;
                case 1: target_dtype = torch::kFloat64; break;
                case 2: target_dtype = torch::kInt32; break;
                case 3: target_dtype = torch::kInt64; break;
                case 4: target_dtype = torch::kInt8; break;
                case 5: target_dtype = torch::kUInt8; break;
                case 6: target_dtype = torch::kBool; break;
                default: target_dtype = torch::kFloat16; break;
            }
            
            auto result2 = torch::empty_like(input_tensor, torch::TensorOptions().dtype(target_dtype));
            
            // Verify dtype was applied
            if (result2.dtype() != target_dtype) {
                std::cerr << "Dtype not applied correctly" << std::endl;
            }
        }

        // Test with requires_grad if we have enough data
        if (offset + 1 < Size) {
            bool requires_grad = (Data[offset++] % 2) == 1;
            
            // Only test requires_grad with floating point tensors
            if (input_tensor.dtype().isFloatingPoint()) {
                auto result3 = torch::empty_like(input_tensor, 
                    torch::TensorOptions().requires_grad(requires_grad));
                
                if (result3.requires_grad() != requires_grad) {
                    std::cerr << "requires_grad not set correctly" << std::endl;
                }
            }
        }

        // Test with different memory format if we have enough data
        if (offset + 1 < Size && input_tensor.dim() >= 4) {
            auto memory_format_choice = Data[offset++] % 3;
            torch::MemoryFormat memory_format;
            
            switch (memory_format_choice) {
                case 0: memory_format = torch::MemoryFormat::Contiguous; break;
                case 1: memory_format = torch::MemoryFormat::ChannelsLast; break;
                default: memory_format = torch::MemoryFormat::Preserve; break;
            }
            
            auto result4 = torch::empty_like(input_tensor, 
                torch::TensorOptions().memory_format(memory_format));
            
            // Basic verification that tensor was created
            if (!result4.defined()) {
                std::cerr << "Result tensor not defined with memory format" << std::endl;
            }
        }

        // Test with device specification if we have enough data
        if (offset + 1 < Size) {
            auto device_choice = Data[offset++] % 2;
            torch::Device target_device = (device_choice == 0) ? torch::kCPU : input_tensor.device();
            
            auto result5 = torch::empty_like(input_tensor, 
                torch::TensorOptions().device(target_device));
            
            if (result5.device() != target_device) {
                std::cerr << "Device not set correctly" << std::endl;
            }
        }

        // Test with multiple options combined
        if (offset + 3 < Size) {
            auto dtype_idx = Data[offset++] % 4;
            bool req_grad = (Data[offset++] % 2) == 1;
            
            torch::ScalarType combined_dtype;
            switch (dtype_idx) {
                case 0: combined_dtype = torch::kFloat32; break;
                case 1: combined_dtype = torch::kFloat64; break;
                case 2: combined_dtype = torch::kInt32; break;
                default: combined_dtype = torch::kInt64; break;
            }
            
            auto options = torch::TensorOptions()
                .dtype(combined_dtype)
                .requires_grad(req_grad && torch::typeMetaToScalarType(torch::dtype(combined_dtype).toScalarType()).isFloatingPoint());
            
            auto result6 = torch::empty_like(input_tensor, options);
            
            // Verify combined options
            if (result6.dtype() != combined_dtype) {
                std::cerr << "Combined dtype not applied" << std::endl;
            }
        }

        // Test edge cases with zero-sized tensors
        if (offset + 1 < Size) {
            auto empty_tensor = torch::empty({0, 5, 0});
            auto result7 = torch::empty_like(empty_tensor);
            
            if (!result7.sizes().equals(empty_tensor.sizes())) {
                std::cerr << "Empty tensor size mismatch" << std::endl;
            }
        }

        // Test with scalar tensor
        if (offset + 1 < Size) {
            auto scalar_tensor = torch::tensor(42.0f);
            auto result8 = torch::empty_like(scalar_tensor);
            
            if (result8.dim() != 0) {
                std::cerr << "Scalar tensor dimension mismatch" << std::endl;
            }
        }

        // Test with very large tensor (if input allows)
        if (offset + 4 < Size && input_tensor.numel() < 1000000) {
            try {
                auto large_sizes = std::vector<int64_t>();
                for (int i = 0; i < std::min(4, (int)(Size - offset)); i++) {
                    int64_t dim_size = (Data[offset + i] % 100) + 1; // 1 to 100
                    large_sizes.push_back(dim_size);
                }
                offset += large_sizes.size();
                
                if (!large_sizes.empty()) {
                    auto large_tensor = torch::zeros(large_sizes);
                    auto result9 = torch::empty_like(large_tensor);
                    
                    if (!result9.sizes().equals(large_tensor.sizes())) {
                        std::cerr << "Large tensor size mismatch" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                // Large tensor creation might fail due to memory constraints
                // This is acceptable behavior
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}