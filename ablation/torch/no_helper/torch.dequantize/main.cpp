#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Test single tensor dequantization
        {
            // Generate quantized tensor parameters
            auto shape = generateRandomShape(Data, Size, offset, 4);
            if (shape.empty()) return 0;

            auto dtype = generateRandomQuantizedDtype(Data, Size, offset);
            auto device = generateRandomDevice(Data, Size, offset);

            // Create a regular tensor first
            torch::Tensor regular_tensor = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            
            // Generate quantization parameters
            double scale = generateRandomFloat(Data, Size, offset, 0.001, 10.0);
            int64_t zero_point = generateRandomInt(Data, Size, offset, -128, 127);
            
            // Create quantized tensor
            torch::Tensor quantized_tensor;
            if (dtype == torch::kQInt8) {
                quantized_tensor = torch::quantize_per_tensor(regular_tensor, scale, zero_point, torch::kQInt8);
            } else if (dtype == torch::kQUInt8) {
                zero_point = generateRandomInt(Data, Size, offset, 0, 255);
                quantized_tensor = torch::quantize_per_tensor(regular_tensor, scale, zero_point, torch::kQUInt8);
            } else if (dtype == torch::kQInt32) {
                quantized_tensor = torch::quantize_per_tensor(regular_tensor, scale, zero_point, torch::kQInt32);
            }

            // Test dequantize
            torch::Tensor dequantized = torch::dequantize(quantized_tensor);
            
            // Verify result properties
            if (!dequantized.dtype().isFloatingPoint()) {
                throw std::runtime_error("Dequantized tensor should be floating point");
            }
            if (!dequantized.sizes().equals(quantized_tensor.sizes())) {
                throw std::runtime_error("Dequantized tensor shape mismatch");
            }
        }

        // Test multiple tensor dequantization
        {
            int num_tensors = generateRandomInt(Data, Size, offset, 1, 5);
            std::vector<torch::Tensor> quantized_tensors;
            
            for (int i = 0; i < num_tensors; i++) {
                auto shape = generateRandomShape(Data, Size, offset, 3);
                if (shape.empty()) continue;

                auto dtype = generateRandomQuantizedDtype(Data, Size, offset);
                auto device = generateRandomDevice(Data, Size, offset);

                // Create regular tensor
                torch::Tensor regular_tensor = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
                
                // Generate quantization parameters
                double scale = generateRandomFloat(Data, Size, offset, 0.001, 10.0);
                int64_t zero_point = generateRandomInt(Data, Size, offset, -128, 127);
                
                // Create quantized tensor
                torch::Tensor quantized_tensor;
                if (dtype == torch::kQInt8) {
                    quantized_tensor = torch::quantize_per_tensor(regular_tensor, scale, zero_point, torch::kQInt8);
                } else if (dtype == torch::kQUInt8) {
                    zero_point = generateRandomInt(Data, Size, offset, 0, 255);
                    quantized_tensor = torch::quantize_per_tensor(regular_tensor, scale, zero_point, torch::kQUInt8);
                } else if (dtype == torch::kQInt32) {
                    quantized_tensor = torch::quantize_per_tensor(regular_tensor, scale, zero_point, torch::kQInt32);
                }
                
                quantized_tensors.push_back(quantized_tensor);
            }

            if (!quantized_tensors.empty()) {
                // Test dequantize with vector of tensors
                std::vector<torch::Tensor> dequantized_tensors = torch::dequantize(quantized_tensors);
                
                // Verify results
                if (dequantized_tensors.size() != quantized_tensors.size()) {
                    throw std::runtime_error("Dequantized tensor count mismatch");
                }
                
                for (size_t i = 0; i < dequantized_tensors.size(); i++) {
                    if (!dequantized_tensors[i].dtype().isFloatingPoint()) {
                        throw std::runtime_error("Dequantized tensor should be floating point");
                    }
                    if (!dequantized_tensors[i].sizes().equals(quantized_tensors[i].sizes())) {
                        throw std::runtime_error("Dequantized tensor shape mismatch");
                    }
                }
            }
        }

        // Test edge cases
        {
            // Test with different quantization schemes
            auto shape = generateRandomShape(Data, Size, offset, 3);
            if (!shape.empty()) {
                auto device = generateRandomDevice(Data, Size, offset);
                torch::Tensor regular_tensor = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
                
                // Test per-channel quantization if supported
                if (regular_tensor.dim() >= 2) {
                    int64_t axis = generateRandomInt(Data, Size, offset, 0, regular_tensor.dim() - 1);
                    int64_t num_channels = regular_tensor.size(axis);
                    
                    torch::Tensor scales = torch::rand({num_channels}, torch::TensorOptions().dtype(torch::kFloat32)) * 0.1 + 0.001;
                    torch::Tensor zero_points = torch::randint(-128, 127, {num_channels}, torch::TensorOptions().dtype(torch::kInt));
                    
                    torch::Tensor per_channel_quantized = torch::quantize_per_channel(regular_tensor, scales, zero_points, axis, torch::kQInt8);
                    torch::Tensor per_channel_dequantized = torch::dequantize(per_channel_quantized);
                    
                    if (!per_channel_dequantized.dtype().isFloatingPoint()) {
                        throw std::runtime_error("Per-channel dequantized tensor should be floating point");
                    }
                }
            }
        }

        // Test with extreme values
        {
            auto shape = generateRandomShape(Data, Size, offset, 2);
            if (!shape.empty()) {
                auto device = generateRandomDevice(Data, Size, offset);
                
                // Test with very small scale
                torch::Tensor regular_tensor = torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat32).device(device));
                double small_scale = 1e-6;
                int64_t zero_point = 0;
                
                torch::Tensor quantized_small_scale = torch::quantize_per_tensor(regular_tensor, small_scale, zero_point, torch::kQInt8);
                torch::Tensor dequantized_small_scale = torch::dequantize(quantized_small_scale);
                
                // Test with large scale
                double large_scale = 100.0;
                torch::Tensor quantized_large_scale = torch::quantize_per_tensor(regular_tensor, large_scale, zero_point, torch::kQInt8);
                torch::Tensor dequantized_large_scale = torch::dequantize(quantized_large_scale);
                
                // Verify both results are floating point
                if (!dequantized_small_scale.dtype().isFloatingPoint() || !dequantized_large_scale.dtype().isFloatingPoint()) {
                    throw std::runtime_error("Extreme scale dequantized tensors should be floating point");
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