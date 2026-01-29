#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::abs

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point for RMSNorm
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get normalized_shape from the last dimension of the input tensor
        std::vector<int64_t> normalized_shape;
        if (input.dim() > 0) {
            int64_t last_dim = input.size(-1);
            if (last_dim > 0) {
                normalized_shape.push_back(last_dim);
            } else {
                normalized_shape.push_back(1);
            }
        } else {
            // For scalar tensors, use a default shape
            normalized_shape.push_back(1);
        }
        
        // Extract epsilon parameter from the input data
        double epsilon = 1e-5; // Default value
        if (offset + sizeof(float) <= Size) {
            float eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure epsilon is positive and reasonable
            if (std::isfinite(eps_raw) && eps_raw > 0 && eps_raw < 1.0f) {
                epsilon = static_cast<double>(eps_raw);
            } else if (std::isfinite(eps_raw) && eps_raw < 0 && eps_raw > -1.0f) {
                epsilon = static_cast<double>(std::abs(eps_raw));
            }
            // Otherwise keep default
        }
        
        // Apply RMSNorm using torch::rms_norm
        try {
            torch::Tensor output = torch::rms_norm(input, normalized_shape, torch::nullopt, epsilon);
        } catch (const c10::Error&) {
            // Expected failures for invalid shapes/types
        }
        
        // Test with different weight configurations
        if (offset + 1 <= Size) {
            bool use_weight = Data[offset++] & 1;
            
            if (use_weight && input.dim() > 0 && normalized_shape[0] > 0) {
                try {
                    // Create a weight tensor with the same shape as normalized_shape
                    torch::Tensor weight = torch::ones(normalized_shape, input.options());
                    
                    // Optionally scale weight based on fuzzer data
                    if (offset + sizeof(float) <= Size) {
                        float scale;
                        std::memcpy(&scale, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        if (std::isfinite(scale)) {
                            weight = weight * scale;
                        }
                    }
                    
                    // Apply RMSNorm with weight
                    torch::Tensor output_with_weight = torch::rms_norm(input, normalized_shape, weight, epsilon);
                } catch (const c10::Error&) {
                    // Expected failures
                }
            }
        }
        
        // Test with multi-dimensional normalized_shape
        if (offset + 1 <= Size && input.dim() >= 2) {
            uint8_t num_dims = (Data[offset++] % std::min(input.dim(), static_cast<int64_t>(3))) + 1;
            
            try {
                std::vector<int64_t> multi_normalized_shape;
                for (int64_t i = input.dim() - num_dims; i < input.dim(); i++) {
                    multi_normalized_shape.push_back(input.size(i));
                }
                
                if (!multi_normalized_shape.empty()) {
                    torch::Tensor output_multi = torch::rms_norm(input, multi_normalized_shape, torch::nullopt, epsilon);
                }
            } catch (const c10::Error&) {
                // Expected failures for invalid shapes
            }
        }
        
        // Test with different data types if there's enough data left
        if (offset + 1 <= Size && input.dim() > 0) {
            uint8_t dtype_selector = Data[offset++];
            
            try {
                // Only test with floating point types
                torch::ScalarType dtype;
                switch (dtype_selector % 3) {
                    case 0: dtype = torch::kFloat32; break;
                    case 1: dtype = torch::kFloat64; break;
                    case 2: dtype = torch::kFloat16; break;
                    default: dtype = torch::kFloat32; break;
                }
                
                // Convert input to the new data type
                torch::Tensor input_converted = input.to(dtype);
                
                // Apply RMSNorm to the converted input
                torch::Tensor output_dtype = torch::rms_norm(input_converted, normalized_shape, torch::nullopt, epsilon);
            } catch (const c10::Error&) {
                // Expected failures for unsupported dtypes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}