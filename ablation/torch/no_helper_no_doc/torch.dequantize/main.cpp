#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions
        auto dims = extract_tensor_dims(Data, Size, offset, 4); // Max 4D tensor
        if (dims.empty()) {
            return 0;
        }

        // Extract quantization parameters
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Extract scale (float)
        float scale;
        std::memcpy(&scale, Data + offset, sizeof(float));
        offset += sizeof(float);
        
        // Clamp scale to reasonable range to avoid overflow/underflow
        scale = std::max(1e-6f, std::min(1e6f, std::abs(scale)));
        
        // Extract zero_point (int32)
        int32_t zero_point;
        std::memcpy(&zero_point, Data + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
        
        // Clamp zero_point to valid range for different dtypes
        zero_point = std::max(-128, std::min(127, zero_point));

        // Extract quantization scheme and dtype flags
        if (offset >= Size) {
            return 0;
        }
        
        uint8_t flags = Data[offset++];
        bool per_channel = (flags & 0x01) != 0;
        bool use_quint8 = (flags & 0x02) != 0;
        bool use_qint8 = (flags & 0x04) != 0;
        bool use_qint32 = (flags & 0x08) != 0;

        // Default to quint8 if no specific dtype selected
        torch::ScalarType qtype = torch::kQUInt8;
        if (use_qint8) {
            qtype = torch::kQInt8;
        } else if (use_qint32) {
            qtype = torch::kQInt32;
            zero_point = 0; // qint32 typically has zero_point = 0
        }

        // Calculate total elements
        int64_t total_elements = 1;
        for (auto dim : dims) {
            total_elements *= dim;
        }

        // Limit tensor size to prevent memory issues
        if (total_elements > 10000) {
            return 0;
        }

        // Create quantized tensor
        torch::Tensor qtensor;
        
        if (per_channel && dims.size() > 1) {
            // Per-channel quantization
            int64_t num_channels = dims[0]; // Use first dimension as channel dimension
            
            // Create per-channel scales and zero_points
            auto scales = torch::full({num_channels}, scale, torch::kFloat);
            auto zero_points = torch::full({num_channels}, zero_point, torch::kLong);
            
            // Add some variation to scales and zero_points
            if (offset + num_channels * 2 <= Size) {
                for (int64_t i = 0; i < num_channels && offset < Size; ++i) {
                    if (offset < Size) {
                        float scale_variation = static_cast<float>(Data[offset++]) / 255.0f * 0.1f + 0.95f;
                        scales[i] = scale * scale_variation;
                    }
                    if (offset < Size) {
                        int zp_variation = static_cast<int>(Data[offset++]) % 21 - 10; // -10 to +10
                        zero_points[i] = std::max(-128L, std::min(127L, static_cast<long>(zero_point + zp_variation)));
                    }
                }
            }
            
            // Create random integer data
            auto int_data = create_random_tensor(dims, torch::kInt8, Data, Size, offset);
            
            // Create per-channel quantized tensor
            qtensor = torch::_make_per_channel_quantized_tensor(
                int_data, scales, zero_points, 0, qtype);
        } else {
            // Per-tensor quantization
            // Create random integer data based on quantization type
            torch::ScalarType data_type = torch::kUInt8;
            if (qtype == torch::kQInt8) {
                data_type = torch::kInt8;
            } else if (qtype == torch::kQInt32) {
                data_type = torch::kInt32;
            }
            
            auto int_data = create_random_tensor(dims, data_type, Data, Size, offset);
            
            // Create per-tensor quantized tensor
            qtensor = torch::_make_per_tensor_quantized_tensor(
                int_data, scale, zero_point, qtype);
        }

        // Test dequantize operation
        torch::Tensor dequantized = torch::dequantize(qtensor);
        
        // Verify the result
        if (!dequantized.defined()) {
            std::cerr << "Dequantize returned undefined tensor" << std::endl;
            return -1;
        }
        
        // Check that output is float type
        if (dequantized.scalar_type() != torch::kFloat) {
            std::cerr << "Dequantized tensor should be float type" << std::endl;
            return -1;
        }
        
        // Check dimensions are preserved
        if (dequantized.sizes() != qtensor.sizes()) {
            std::cerr << "Dequantized tensor dimensions don't match input" << std::endl;
            return -1;
        }
        
        // Test with different tensor configurations
        if (offset < Size) {
            // Test with contiguous tensor
            auto contiguous_qtensor = qtensor.contiguous();
            auto contiguous_dequantized = torch::dequantize(contiguous_qtensor);
            
            // Test with non-contiguous tensor (if possible)
            if (qtensor.dim() > 1) {
                try {
                    auto transposed_qtensor = qtensor.transpose(0, 1);
                    auto transposed_dequantized = torch::dequantize(transposed_qtensor);
                } catch (...) {
                    // Some quantized tensors might not support transpose
                }
            }
        }
        
        // Test edge cases
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Test with zero-sized tensor
            if (edge_case & 0x01) {
                try {
                    auto empty_dims = std::vector<int64_t>{0};
                    auto empty_data = torch::empty(empty_dims, torch::kUInt8);
                    auto empty_qtensor = torch::_make_per_tensor_quantized_tensor(
                        empty_data, scale, zero_point, qtype);
                    auto empty_dequantized = torch::dequantize(empty_qtensor);
                } catch (...) {
                    // Empty tensors might not be supported
                }
            }
            
            // Test with single element tensor
            if (edge_case & 0x02) {
                auto single_dims = std::vector<int64_t>{1};
                auto single_data = torch::randint(0, 255, single_dims, torch::kUInt8);
                auto single_qtensor = torch::_make_per_tensor_quantized_tensor(
                    single_data, scale, zero_point, qtype);
                auto single_dequantized = torch::dequantize(single_qtensor);
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