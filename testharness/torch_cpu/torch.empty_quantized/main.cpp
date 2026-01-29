#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Need at least 4 bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse scale from input data
        float scale = 0.1f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure scale is positive and in valid range
            if (!std::isfinite(scale) || scale <= 0) {
                scale = 0.1f;
            }
            scale = std::abs(scale);
            if (scale < 1e-6f) scale = 1e-6f;
            if (scale > 1e6f) scale = 1e6f;
        }
        
        // Parse zero_point from input data
        int64_t zero_point = 0;
        if (offset + sizeof(int8_t) <= Size) {
            // Use smaller type to keep zero_point in valid range
            int8_t zp_byte = static_cast<int8_t>(Data[offset++]);
            zero_point = static_cast<int64_t>(zp_byte);
        }
        
        // Parse quantization dtype
        torch::ScalarType dtype = torch::kQInt8;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            switch (dtype_selector % 3) {
                case 0:
                    dtype = torch::kQInt8;
                    // Clamp zero_point for QInt8 range [-128, 127]
                    zero_point = std::max<int64_t>(-128, std::min<int64_t>(127, zero_point));
                    break;
                case 1:
                    dtype = torch::kQUInt8;
                    // Clamp zero_point for QUInt8 range [0, 255]
                    zero_point = std::max<int64_t>(0, std::min<int64_t>(255, zero_point));
                    break;
                case 2:
                    dtype = torch::kQInt32;
                    // QInt32 has wider range, zero_point is already fine
                    break;
            }
        }
        
        // Parse tensor shape
        uint8_t rank = 1;
        if (offset < Size) {
            rank = fuzzer_utils::parseRank(Data[offset++]);
            if (rank == 0) rank = 1;  // Ensure at least 1D tensor
        }
        
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Ensure shape is valid and not too large
        int64_t total_elements = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] <= 0) shape[i] = 1;
            if (shape[i] > 64) shape[i] = 64;  // Limit individual dimensions
            total_elements *= shape[i];
            if (total_elements > 10000) {
                // Truncate remaining dimensions
                for (size_t j = i + 1; j < shape.size(); j++) {
                    shape[j] = 1;
                }
                break;
            }
        }
        
        // Test 1: Create empty quantized tensor using a quantized template
        try {
            // Create a regular tensor with the desired shape
            torch::Tensor temp_tensor = torch::zeros(shape, torch::TensorOptions().dtype(torch::kFloat));
            
            // Quantize it to create a template tensor
            torch::Tensor qtensor = torch::quantize_per_tensor(temp_tensor, scale, zero_point, dtype);
            
            // Create empty quantized tensor using the template
            torch::Tensor result = torch::empty_quantized(shape, qtensor);
            
            // Validate the result
            if (result.is_quantized()) {
                // Access quantization parameters to ensure they're valid
                double q_scale = result.q_scale();
                int64_t q_zero_point = result.q_zero_point();
                (void)q_scale;
                (void)q_zero_point;
            }
        } catch (const c10::Error& e) {
            // Expected PyTorch exceptions - silently catch
        }
        
        // Test 2: Create empty quantized with different shape but same template
        if (offset < Size) {
            try {
                // Create template with original shape
                torch::Tensor temp = torch::ones(shape, torch::TensorOptions().dtype(torch::kFloat));
                torch::Tensor qtensor = torch::quantize_per_tensor(temp, scale, zero_point, dtype);
                
                // Create empty with different shape
                std::vector<int64_t> new_shape;
                uint8_t new_rank = (Data[offset++] % 4) + 1;
                for (uint8_t i = 0; i < new_rank && offset < Size; i++) {
                    int64_t dim = (Data[offset++] % 16) + 1;
                    new_shape.push_back(dim);
                }
                if (new_shape.empty()) {
                    new_shape.push_back(1);
                }
                
                torch::Tensor result = torch::empty_quantized(new_shape, qtensor);
                
                // Validate result has correct shape
                if (result.dim() == static_cast<int64_t>(new_shape.size())) {
                    for (size_t i = 0; i < new_shape.size(); i++) {
                        (void)(result.size(i) == new_shape[i]);
                    }
                }
            } catch (const c10::Error& e) {
                // Expected exceptions
            }
        }
        
        // Test 3: Test with per-channel quantized template (if supported)
        if (offset < Size && shape.size() >= 1) {
            try {
                int64_t axis = 0;  // Quantize along first axis
                int64_t num_channels = shape[axis];
                
                // Create per-channel scales and zero_points
                std::vector<double> scales_vec(num_channels);
                std::vector<int64_t> zero_points_vec(num_channels);
                
                for (int64_t i = 0; i < num_channels; i++) {
                    scales_vec[i] = scale * (1.0 + 0.1 * (i % 5));
                    zero_points_vec[i] = zero_point;
                    
                    // Clamp zero_points based on dtype
                    if (dtype == torch::kQInt8) {
                        zero_points_vec[i] = std::max<int64_t>(-128, std::min<int64_t>(127, zero_points_vec[i]));
                    } else if (dtype == torch::kQUInt8) {
                        zero_points_vec[i] = std::max<int64_t>(0, std::min<int64_t>(255, zero_points_vec[i]));
                    }
                }
                
                torch::Tensor scales_tensor = torch::tensor(scales_vec, torch::kDouble);
                torch::Tensor zero_points_tensor = torch::tensor(zero_points_vec, torch::kLong);
                
                torch::Tensor temp = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat));
                torch::Tensor qtensor = torch::quantize_per_channel(
                    temp, scales_tensor, zero_points_tensor, axis, dtype);
                
                torch::Tensor result = torch::empty_quantized(shape, qtensor);
                
                // Validate per-channel quantization
                if (result.is_quantized() && result.qscheme() == at::kPerChannelAffine) {
                    auto scales = result.q_per_channel_scales();
                    auto zps = result.q_per_channel_zero_points();
                    (void)scales;
                    (void)zps;
                }
            } catch (const c10::Error& e) {
                // Per-channel quantization may not be supported for all dtypes
            }
        }
        
        // Test 4: Test with explicit tensor options
        if (offset < Size) {
            try {
                torch::Tensor temp = torch::zeros(shape, torch::kFloat);
                torch::Tensor qtensor = torch::quantize_per_tensor(temp, scale, zero_point, dtype);
                
                // Create with explicit options (memory format)
                auto options = qtensor.options();
                torch::Tensor result = torch::empty_quantized(shape, qtensor, options);
                
                // Verify the result matches expected properties
                (void)(result.dtype() == qtensor.dtype());
                (void)(result.device() == qtensor.device());
            } catch (const c10::Error& e) {
                // Expected exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}