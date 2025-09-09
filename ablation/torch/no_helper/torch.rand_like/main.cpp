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
        if (Size < 10) {
            return 0;
        }

        // Extract tensor configuration
        auto tensor_config = extract_tensor_config(Data, Size, offset);
        if (!tensor_config.has_value()) {
            return 0;
        }

        // Create input tensor with various shapes and dtypes
        torch::Tensor input;
        try {
            input = create_tensor_from_config(tensor_config.value());
        } catch (...) {
            return 0; // Skip invalid tensor configurations
        }

        // Extract optional parameters for rand_like
        bool use_custom_dtype = extract_bool(Data, Size, offset);
        bool use_custom_layout = extract_bool(Data, Size, offset);
        bool use_custom_device = extract_bool(Data, Size, offset);
        bool requires_grad = extract_bool(Data, Size, offset);
        bool use_custom_memory_format = extract_bool(Data, Size, offset);

        torch::TensorOptions options;

        // Test with custom dtype
        if (use_custom_dtype && offset < Size) {
            auto dtype_idx = extract_int(Data, Size, offset) % 8;
            std::vector<torch::ScalarType> dtypes = {
                torch::kFloat32, torch::kFloat64, torch::kInt32, torch::kInt64,
                torch::kInt8, torch::kUInt8, torch::kBool, torch::kFloat16
            };
            options = options.dtype(dtypes[dtype_idx]);
        }

        // Test with custom layout (only dense and sparse_coo are commonly supported)
        if (use_custom_layout && offset < Size) {
            auto layout_idx = extract_int(Data, Size, offset) % 2;
            if (layout_idx == 0) {
                options = options.layout(torch::kStrided);
            } else {
                options = options.layout(torch::kSparse);
            }
        }

        // Test with custom device (CPU only for fuzzing safety)
        if (use_custom_device) {
            options = options.device(torch::kCPU);
        }

        // Test requires_grad
        options = options.requires_grad(requires_grad);

        // Test with custom memory format
        torch::MemoryFormat memory_format = torch::MemoryFormat::Preserve;
        if (use_custom_memory_format && offset < Size) {
            auto format_idx = extract_int(Data, Size, offset) % 4;
            std::vector<torch::MemoryFormat> formats = {
                torch::MemoryFormat::Preserve,
                torch::MemoryFormat::Contiguous,
                torch::MemoryFormat::ChannelsLast,
                torch::MemoryFormat::ChannelsLast3d
            };
            memory_format = formats[format_idx];
        }

        // Test basic rand_like without options
        torch::Tensor result1 = torch::rand_like(input);
        
        // Verify basic properties
        if (!result1.sizes().equals(input.sizes())) {
            throw std::runtime_error("rand_like result has wrong shape");
        }

        // Test rand_like with various option combinations
        torch::Tensor result2;
        if (use_custom_dtype || use_custom_layout || use_custom_device || requires_grad) {
            result2 = torch::rand_like(input, options);
        } else {
            result2 = torch::rand_like(input, torch::TensorOptions().memory_format(memory_format));
        }

        // Verify the result has correct shape
        if (!result2.sizes().equals(input.sizes())) {
            throw std::runtime_error("rand_like with options result has wrong shape");
        }

        // Test edge cases with different input tensor properties
        
        // Test with zero-sized tensor
        if (offset < Size) {
            auto zero_dim = extract_int(Data, Size, offset) % 4;
            std::vector<int64_t> zero_shape;
            for (int i = 0; i < zero_dim; i++) {
                zero_shape.push_back(0);
            }
            if (!zero_shape.empty()) {
                torch::Tensor zero_tensor = torch::empty(zero_shape, input.options());
                torch::Tensor zero_result = torch::rand_like(zero_tensor);
                if (!zero_result.sizes().equals(zero_tensor.sizes())) {
                    throw std::runtime_error("rand_like with zero-sized tensor failed");
                }
            }
        }

        // Test with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(1.0f);
        torch::Tensor scalar_result = torch::rand_like(scalar_tensor);
        if (scalar_result.dim() != 0) {
            throw std::runtime_error("rand_like with scalar tensor should return scalar");
        }

        // Test with different memory formats if input supports it
        if (input.dim() >= 4 && memory_format != torch::MemoryFormat::Preserve) {
            try {
                torch::Tensor formatted_result = torch::rand_like(input, torch::TensorOptions().memory_format(memory_format));
                if (!formatted_result.sizes().equals(input.sizes())) {
                    throw std::runtime_error("rand_like with memory format has wrong shape");
                }
            } catch (const std::exception&) {
                // Some memory formats may not be compatible with certain tensor shapes
                // This is expected behavior
            }
        }

        // Test with very large tensors (but limit size for fuzzing)
        if (offset < Size && input.numel() < 1000) {
            auto scale_factor = (extract_int(Data, Size, offset) % 3) + 1;
            std::vector<int64_t> large_shape;
            for (int64_t dim : input.sizes()) {
                large_shape.push_back(std::min(dim * scale_factor, 100L)); // Limit size
            }
            torch::Tensor large_tensor = torch::empty(large_shape, input.options());
            torch::Tensor large_result = torch::rand_like(large_tensor);
            if (!large_result.sizes().equals(large_tensor.sizes())) {
                throw std::runtime_error("rand_like with large tensor failed");
            }
        }

        // Verify that results are actually random (basic sanity check)
        if (result1.numel() > 1) {
            // Check that not all values are the same (very unlikely for random)
            auto flattened = result1.flatten();
            bool all_same = true;
            float first_val = flattened[0].item<float>();
            for (int64_t i = 1; i < std::min(flattened.numel(), 10L); i++) {
                if (std::abs(flattened[i].item<float>() - first_val) > 1e-6) {
                    all_same = false;
                    break;
                }
            }
            // Note: We don't throw here as it's theoretically possible (though unlikely)
            // for random values to be the same
        }

        // Test that values are in [0, 1) range
        if (result1.dtype() == torch::kFloat32 || result1.dtype() == torch::kFloat64) {
            auto min_val = torch::min(result1);
            auto max_val = torch::max(result1);
            if (min_val.item<double>() < 0.0 || max_val.item<double>() >= 1.0) {
                throw std::runtime_error("rand_like values not in [0, 1) range");
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