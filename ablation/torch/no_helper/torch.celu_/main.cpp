#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 16) {
            // Need minimum bytes for basic tensor construction
            return 0;
        }

        size_t offset = 0;

        // Extract parameters for tensor construction
        uint8_t rank = Data[offset++] % 5;  // Limit rank to 0-4 for practical fuzzing
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (uint8_t i = 0; i < rank && offset < Size; i++) {
            int64_t dim = static_cast<int64_t>(Data[offset++] % 10);  // Small dims including 0
            shape.push_back(dim);
        }

        // Determine dtype
        uint8_t dtype_selector = offset < Size ? Data[offset++] % 4 : 0;
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kFloat16; break;
            default: dtype = torch::kBFloat16; break;
        }

        // Extract alpha parameter
        float alpha_raw = 1.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&alpha_raw, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Handle special float values
        if (std::isnan(alpha_raw) || std::isinf(alpha_raw)) {
            // Keep these values to test edge cases
        } else {
            // Bound alpha to reasonable range
            alpha_raw = std::fmod(alpha_raw, 100.0f);
        }
        double alpha = static_cast<double>(alpha_raw);

        // Determine if tensor should require grad
        bool requires_grad = offset < Size ? (Data[offset++] % 2 == 1) : false;

        // Create tensor with various configurations
        torch::Tensor input;
        
        // Calculate total elements
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }

        if (total_elements == 0) {
            // Create empty tensor
            input = torch::empty(shape, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
        } else if (total_elements > 0 && total_elements <= 10000) {
            // Create tensor with data from fuzzer input
            uint8_t init_type = offset < Size ? Data[offset++] % 6 : 0;
            
            switch (init_type) {
                case 0:
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
                    break;
                case 1:
                    input = torch::zeros(shape, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
                    break;
                case 2:
                    input = torch::ones(shape, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
                    break;
                case 3:
                    // Fill with specific value
                    {
                        float fill_val = offset + sizeof(float) <= Size ? 
                            *reinterpret_cast<const float*>(Data + offset) : 0.5f;
                        input = torch::full(shape, fill_val, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
                    }
                    break;
                case 4:
                    // Create with negative values
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad)) - 5.0;
                    break;
                default:
                    // Create with mixed positive/negative values
                    input = torch::randn(shape, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad)) * 10.0;
                    break;
            }
        } else {
            // Too many elements, create smaller tensor
            input = torch::randn({2, 2}, torch::TensorOptions().dtype(dtype).requires_grad(requires_grad));
        }

        // Test with non-contiguous tensors sometimes
        if (offset < Size && Data[offset++] % 4 == 0 && input.dim() >= 2) {
            input = input.transpose(0, input.dim() - 1);
        }

        // Apply celu_ in-place operation
        torch::Tensor result = input.celu_(alpha);
        
        // Verify in-place operation worked (result should be same as input)
        if (result.data_ptr() != input.data_ptr()) {
            // This shouldn't happen for in-place operation
            return -1;
        }

        // Try to trigger additional code paths
        if (requires_grad && input.requires_grad()) {
            // Compute gradient to test backward pass
            torch::Tensor grad_output = torch::ones_like(input);
            input.backward(grad_output);
        }

        // Access result to ensure computation completed
        if (input.numel() > 0) {
            auto first_elem = input.flatten()[0].item<float>();
            (void)first_elem;  // Suppress unused warning
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}