#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract reshape parameters from the remaining data
        std::vector<int64_t> new_shape;
        
        // Determine number of dimensions for reshape
        if (offset < Size) {
            uint8_t num_dims = (Data[offset++] % 5) + 1; // 1-5 dimensions (avoid 0)
            
            // Parse each dimension
            for (uint8_t i = 0; i < num_dims && offset + sizeof(int32_t) <= Size; ++i) {
                int32_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int32_t));
                offset += sizeof(int32_t);
                
                // Constrain dimension values to reasonable range
                // Allow -1 for auto-infer, and positive values up to 1024
                if (dim_value < -1) {
                    dim_value = -1;
                } else if (dim_value > 1024) {
                    dim_value = dim_value % 1024 + 1;
                } else if (dim_value == 0) {
                    dim_value = 1; // Avoid 0 dimensions (except for empty tensors)
                }
                
                new_shape.push_back(static_cast<int64_t>(dim_value));
            }
            
            // If we have at least one dimension, possibly use -1 for auto-infer
            if (offset < Size && new_shape.size() > 0) {
                uint8_t use_special = Data[offset++] % 2;
                if (use_special) {
                    // Replace a random dimension with -1 (auto-infer)
                    size_t idx = 0;
                    if (offset < Size) {
                        idx = Data[offset++] % new_shape.size();
                    }
                    new_shape[idx] = -1;
                }
            }
        }
        
        // If we couldn't extract any shape, use some defaults
        if (new_shape.empty()) {
            if (offset < Size) {
                uint8_t pattern = Data[offset++] % 4;
                switch (pattern) {
                    case 0: // Flatten
                        new_shape.push_back(-1);
                        break;
                    case 1: // Make 2D
                        new_shape.push_back(-1);
                        new_shape.push_back(1);
                        break;
                    case 2: // Make 3D
                        new_shape.push_back(-1);
                        new_shape.push_back(1);
                        new_shape.push_back(1);
                        break;
                    case 3: // Keep original or flatten
                        if (input_tensor.numel() == 0) {
                            new_shape.push_back(0);
                        } else {
                            new_shape.push_back(-1);
                        }
                        break;
                }
            } else {
                // Default to flatten
                new_shape.push_back(-1);
            }
        }
        
        // Apply reshape operation
        torch::Tensor output;
        
        // Try different reshape variants
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 3;
        }
        
        switch (variant) {
            case 0:
                // Standard reshape (method)
                output = input_tensor.reshape(new_shape);
                break;
            case 1:
                // View (when possible) - stricter, requires contiguous memory
                try {
                    output = input_tensor.view(new_shape);
                } catch (const std::exception&) {
                    // Fallback to reshape if view fails (expected for non-contiguous)
                    output = input_tensor.reshape(new_shape);
                }
                break;
            case 2:
                // Functional API
                output = torch::reshape(input_tensor, new_shape);
                break;
        }
        
        // Basic validation - number of elements should match
        // Note: This should always be true for valid reshapes
        if (input_tensor.numel() != output.numel()) {
            throw std::runtime_error("Element count mismatch after reshape");
        }
        
        // Force evaluation
        if (output.numel() > 0) {
            output.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}