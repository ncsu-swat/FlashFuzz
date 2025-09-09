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

        // Extract basic parameters
        auto dtype_idx = extract_int(Data, Size, offset) % 12; // 12 common dtypes
        auto device_idx = extract_int(Data, Size, offset) % 2; // CPU or CUDA
        auto layout_idx = extract_int(Data, Size, offset) % 2; // strided or sparse_coo
        auto requires_grad = extract_bool(Data, Size, offset);
        auto pin_memory = extract_bool(Data, Size, offset);
        auto memory_format_idx = extract_int(Data, Size, offset) % 4; // 4 memory formats

        // Create input tensor with various shapes and properties
        auto input_shape = extract_tensor_shape(Data, Size, offset, 1, 6); // 1-6 dimensions
        
        // Map indices to actual PyTorch types
        std::vector<torch::Dtype> dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kInt32, torch::kInt64,
            torch::kInt8, torch::kInt16, torch::kUInt8, torch::kBool,
            torch::kFloat16, torch::kBFloat16, torch::kComplexFloat, torch::kComplexDouble
        };
        
        std::vector<torch::Device> devices = {
            torch::kCPU,
            torch::kCUDA
        };
        
        std::vector<torch::Layout> layouts = {
            torch::kStrided,
            torch::kSparse
        };
        
        std::vector<torch::MemoryFormat> memory_formats = {
            torch::MemoryFormat::Contiguous,
            torch::MemoryFormat::Preserve,
            torch::MemoryFormat::ChannelsLast,
            torch::MemoryFormat::ChannelsLast3d
        };

        auto dtype = dtypes[dtype_idx];
        auto device = devices[device_idx];
        auto layout = layouts[layout_idx];
        auto memory_format = memory_formats[memory_format_idx];

        // Skip CUDA if not available
        if (device.is_cuda() && !torch::cuda::is_available()) {
            device = torch::kCPU;
        }

        // Create input tensor
        torch::TensorOptions input_options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .layout(layout)
            .requires_grad(requires_grad);

        // Skip sparse layout for simplicity in fuzzing
        if (layout == torch::kSparse) {
            input_options = input_options.layout(torch::kStrided);
        }

        torch::Tensor input;
        try {
            input = torch::randn(input_shape, input_options);
        } catch (...) {
            // If tensor creation fails, try with simpler options
            input = torch::randn(input_shape, torch::kFloat32);
        }

        // Test 1: Basic empty_like
        auto result1 = torch::empty_like(input);
        
        // Verify basic properties
        if (result1.sizes() != input.sizes()) {
            throw std::runtime_error("Shape mismatch in basic empty_like");
        }
        if (result1.dtype() != input.dtype()) {
            throw std::runtime_error("Dtype mismatch in basic empty_like");
        }

        // Test 2: empty_like with different dtype
        if (offset < Size - 4) {
            auto new_dtype_idx = extract_int(Data, Size, offset) % dtypes.size();
            auto new_dtype = dtypes[new_dtype_idx];
            
            try {
                auto result2 = torch::empty_like(input, new_dtype);
                if (result2.sizes() != input.sizes()) {
                    throw std::runtime_error("Shape mismatch in dtype-specified empty_like");
                }
                if (result2.dtype() != new_dtype) {
                    throw std::runtime_error("Dtype mismatch in dtype-specified empty_like");
                }
            } catch (const c10::Error&) {
                // Some dtype conversions might not be supported, ignore
            }
        }

        // Test 3: empty_like with TensorOptions
        if (offset < Size - 8) {
            auto new_device_idx = extract_int(Data, Size, offset) % 2;
            auto new_requires_grad = extract_bool(Data, Size, offset);
            auto new_pin_memory = extract_bool(Data, Size, offset);
            
            auto new_device = devices[new_device_idx];
            if (new_device.is_cuda() && !torch::cuda::is_available()) {
                new_device = torch::kCPU;
            }

            torch::TensorOptions options = torch::TensorOptions()
                .device(new_device)
                .requires_grad(new_requires_grad);
            
            // Only set pin_memory for CPU tensors
            if (new_device.is_cpu()) {
                options = options.pinned_memory(new_pin_memory);
            }

            try {
                auto result3 = torch::empty_like(input, options);
                if (result3.sizes() != input.sizes()) {
                    throw std::runtime_error("Shape mismatch in options-specified empty_like");
                }
                if (result3.device() != new_device) {
                    throw std::runtime_error("Device mismatch in options-specified empty_like");
                }
            } catch (const c10::Error&) {
                // Some option combinations might not be supported, ignore
            }
        }

        // Test 4: empty_like with memory format
        if (offset < Size - 4) {
            auto new_memory_format_idx = extract_int(Data, Size, offset) % memory_formats.size();
            auto new_memory_format = memory_formats[new_memory_format_idx];
            
            try {
                auto result4 = torch::empty_like(input, torch::TensorOptions(), new_memory_format);
                if (result4.sizes() != input.sizes()) {
                    throw std::runtime_error("Shape mismatch in memory-format-specified empty_like");
                }
            } catch (const c10::Error&) {
                // Some memory format combinations might not be supported, ignore
            }
        }

        // Test 5: Test with edge case tensors
        if (offset < Size - 4) {
            auto edge_case = extract_int(Data, Size, offset) % 4;
            torch::Tensor edge_input;
            
            try {
                switch (edge_case) {
                    case 0: // Empty tensor
                        edge_input = torch::empty({0}, input_options);
                        break;
                    case 1: // Scalar tensor
                        edge_input = torch::scalar_tensor(1.0, input_options);
                        break;
                    case 2: // Large tensor (if memory allows)
                        if (input_shape.size() == 1 && input_shape[0] < 1000) {
                            edge_input = torch::randn({1000}, input_options);
                        } else {
                            edge_input = input;
                        }
                        break;
                    case 3: // High dimensional tensor
                        edge_input = torch::randn({2, 2, 2, 2, 2}, input_options);
                        break;
                    default:
                        edge_input = input;
                }
                
                auto result5 = torch::empty_like(edge_input);
                if (result5.sizes() != edge_input.sizes()) {
                    throw std::runtime_error("Shape mismatch in edge case empty_like");
                }
                
            } catch (const c10::Error&) {
                // Edge cases might fail due to memory or other constraints, ignore
            } catch (const std::bad_alloc&) {
                // Memory allocation might fail for large tensors, ignore
            }
        }

        // Test 6: Test with different input tensor states
        if (offset < Size - 4) {
            auto state_test = extract_int(Data, Size, offset) % 3;
            torch::Tensor state_input = input.clone();
            
            try {
                switch (state_test) {
                    case 0: // Transposed tensor
                        if (state_input.dim() >= 2) {
                            state_input = state_input.transpose(0, 1);
                        }
                        break;
                    case 1: // Sliced tensor
                        if (state_input.numel() > 1) {
                            state_input = state_input.slice(0, 0, std::min(2L, state_input.size(0)));
                        }
                        break;
                    case 2: // Reshaped tensor
                        if (state_input.numel() >= 4) {
                            state_input = state_input.view({-1});
                        }
                        break;
                }
                
                auto result6 = torch::empty_like(state_input);
                if (result6.sizes() != state_input.sizes()) {
                    throw std::runtime_error("Shape mismatch in state-modified empty_like");
                }
                
            } catch (const c10::Error&) {
                // Some tensor state modifications might not be compatible, ignore
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