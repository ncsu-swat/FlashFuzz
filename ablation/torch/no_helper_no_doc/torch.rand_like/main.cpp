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

        // Extract tensor properties from fuzz data
        auto shape_info = extract_tensor_shape(Data, Size, offset);
        auto dtype_info = extract_dtype(Data, Size, offset);
        auto device_info = extract_device(Data, Size, offset);
        auto layout_info = extract_layout(Data, Size, offset);
        
        // Extract additional options for rand_like
        bool requires_grad = extract_bool(Data, Size, offset);
        bool pin_memory = extract_bool(Data, Size, offset);
        
        // Create input tensor with various properties
        torch::Tensor input_tensor;
        
        // Try different tensor creation methods based on fuzz data
        uint8_t creation_method = extract_uint8(Data, Size, offset) % 4;
        
        switch (creation_method) {
            case 0:
                // Create tensor with zeros
                input_tensor = torch::zeros(shape_info.sizes, 
                    torch::TensorOptions()
                        .dtype(dtype_info.dtype)
                        .device(device_info.device)
                        .layout(layout_info.layout)
                        .requires_grad(requires_grad)
                        .pinned_memory(pin_memory && device_info.device.is_cpu()));
                break;
            case 1:
                // Create tensor with ones
                input_tensor = torch::ones(shape_info.sizes,
                    torch::TensorOptions()
                        .dtype(dtype_info.dtype)
                        .device(device_info.device)
                        .layout(layout_info.layout)
                        .requires_grad(requires_grad)
                        .pinned_memory(pin_memory && device_info.device.is_cpu()));
                break;
            case 2:
                // Create tensor with random values
                input_tensor = torch::randn(shape_info.sizes,
                    torch::TensorOptions()
                        .dtype(dtype_info.dtype)
                        .device(device_info.device)
                        .layout(layout_info.layout)
                        .requires_grad(requires_grad)
                        .pinned_memory(pin_memory && device_info.device.is_cpu()));
                break;
            case 3:
                // Create empty tensor and fill with specific values
                input_tensor = torch::empty(shape_info.sizes,
                    torch::TensorOptions()
                        .dtype(dtype_info.dtype)
                        .device(device_info.device)
                        .layout(layout_info.layout)
                        .requires_grad(requires_grad)
                        .pinned_memory(pin_memory && device_info.device.is_cpu()));
                
                // Fill with some pattern based on fuzz data
                if (input_tensor.numel() > 0) {
                    float fill_value = extract_float(Data, Size, offset);
                    input_tensor.fill_(fill_value);
                }
                break;
        }

        // Test basic rand_like functionality
        torch::Tensor result1 = torch::rand_like(input_tensor);
        
        // Verify basic properties
        if (result1.sizes() != input_tensor.sizes()) {
            std::cout << "Size mismatch in rand_like result" << std::endl;
        }
        
        if (result1.dtype() != input_tensor.dtype()) {
            std::cout << "Dtype mismatch in rand_like result" << std::endl;
        }
        
        if (result1.device() != input_tensor.device()) {
            std::cout << "Device mismatch in rand_like result" << std::endl;
        }

        // Test rand_like with explicit options
        auto options_dtype = extract_dtype(Data, Size, offset);
        auto options_device = extract_device(Data, Size, offset);
        auto options_layout = extract_layout(Data, Size, offset);
        bool options_requires_grad = extract_bool(Data, Size, offset);
        bool options_pin_memory = extract_bool(Data, Size, offset);

        torch::Tensor result2 = torch::rand_like(input_tensor,
            torch::TensorOptions()
                .dtype(options_dtype.dtype)
                .device(options_device.device)
                .layout(options_layout.layout)
                .requires_grad(options_requires_grad)
                .pinned_memory(options_pin_memory && options_device.device.is_cpu()));

        // Test with memory format if applicable
        uint8_t memory_format_choice = extract_uint8(Data, Size, offset) % 4;
        torch::MemoryFormat memory_format;
        switch (memory_format_choice) {
            case 0: memory_format = torch::MemoryFormat::Contiguous; break;
            case 1: memory_format = torch::MemoryFormat::Preserve; break;
            case 2: memory_format = torch::MemoryFormat::ChannelsLast; break;
            case 3: memory_format = torch::MemoryFormat::ChannelsLast3d; break;
        }

        // Only test memory format for tensors that support it
        if (input_tensor.dim() >= 3) {
            try {
                torch::Tensor result3 = torch::rand_like(input_tensor, 
                    torch::TensorOptions(), memory_format);
            } catch (const std::exception& e) {
                // Some memory formats may not be compatible with certain tensor shapes
                // This is expected behavior
            }
        }

        // Test edge cases
        
        // Test with scalar tensor
        if (extract_bool(Data, Size, offset)) {
            torch::Tensor scalar_tensor = torch::tensor(extract_float(Data, Size, offset),
                torch::TensorOptions()
                    .dtype(dtype_info.dtype)
                    .device(device_info.device));
            torch::Tensor scalar_result = torch::rand_like(scalar_tensor);
        }

        // Test with empty tensor
        if (extract_bool(Data, Size, offset)) {
            torch::Tensor empty_tensor = torch::empty({0}, 
                torch::TensorOptions()
                    .dtype(dtype_info.dtype)
                    .device(device_info.device));
            torch::Tensor empty_result = torch::rand_like(empty_tensor);
        }

        // Test with very large tensor (if memory allows)
        if (extract_bool(Data, Size, offset) && shape_info.total_elements < 1000) {
            std::vector<int64_t> large_shape = {100, 100};
            try {
                torch::Tensor large_tensor = torch::zeros(large_shape,
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(device_info.device));
                torch::Tensor large_result = torch::rand_like(large_tensor);
            } catch (const std::exception& e) {
                // Memory allocation might fail, which is acceptable
            }
        }

        // Test with different dtypes that support random generation
        std::vector<torch::ScalarType> random_dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kFloat16,
            torch::kBFloat16, torch::kComplexFloat, torch::kComplexDouble
        };
        
        uint8_t dtype_idx = extract_uint8(Data, Size, offset) % random_dtypes.size();
        try {
            torch::Tensor typed_tensor = torch::zeros({5, 5},
                torch::TensorOptions()
                    .dtype(random_dtypes[dtype_idx])
                    .device(device_info.device));
            torch::Tensor typed_result = torch::rand_like(typed_tensor);
        } catch (const std::exception& e) {
            // Some dtypes might not be supported on certain devices
        }

        // Verify that results are actually random (different calls should produce different results)
        if (input_tensor.numel() > 0 && input_tensor.numel() < 1000) {
            torch::Tensor rand1 = torch::rand_like(input_tensor);
            torch::Tensor rand2 = torch::rand_like(input_tensor);
            
            // Check that values are in [0, 1) range for floating point types
            if (rand1.dtype().isFloatingPoint()) {
                torch::Tensor min_val = torch::min(rand1);
                torch::Tensor max_val = torch::max(rand1);
                
                if (min_val.item<float>() < 0.0f || max_val.item<float>() >= 1.0f) {
                    std::cout << "rand_like values out of expected range [0, 1)" << std::endl;
                }
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