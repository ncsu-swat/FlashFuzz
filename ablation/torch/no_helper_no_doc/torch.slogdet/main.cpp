#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions and properties
        auto dims = parseDimensions(Data, Size, offset, 2, 5); // 2D to 5D tensors
        if (dims.empty()) return 0;

        auto dtype = parseDtype(Data, Size, offset);
        auto device = parseDevice(Data, Size, offset);

        // Create square tensor for slogdet (required for determinant computation)
        int64_t n = dims.back(); // Use last dimension as square size
        std::vector<int64_t> square_dims(dims.size(), n);
        
        // Test different tensor configurations
        auto tensor = createTensor(square_dims, dtype, device, Data, Size, offset);
        if (!tensor.defined()) return 0;

        // Ensure tensor is square in last two dimensions for slogdet
        if (tensor.dim() >= 2) {
            auto shape = tensor.sizes().vec();
            if (shape[shape.size()-1] != shape[shape.size()-2]) {
                // Make it square by taking minimum of last two dimensions
                int64_t min_dim = std::min(shape[shape.size()-1], shape[shape.size()-2]);
                shape[shape.size()-1] = min_dim;
                shape[shape.size()-2] = min_dim;
                tensor = tensor.slice(-2, 0, min_dim).slice(-1, 0, min_dim);
            }
        }

        // Test basic slogdet
        auto result = torch::slogdet(tensor);
        auto sign = std::get<0>(result);
        auto logabsdet = std::get<1>(result);

        // Verify result properties
        if (sign.defined() && logabsdet.defined()) {
            // Check that sign values are valid (-1, 0, or 1)
            auto sign_data = sign.flatten();
            for (int64_t i = 0; i < sign_data.numel(); ++i) {
                auto val = sign_data[i].item<double>();
                if (std::abs(val) > 1.0 + 1e-6 && val != 0.0) {
                    std::cerr << "Invalid sign value: " << val << std::endl;
                }
            }
        }

        // Test with different tensor types if applicable
        if (tensor.dtype() != torch::kFloat64) {
            try {
                auto double_tensor = tensor.to(torch::kFloat64);
                auto double_result = torch::slogdet(double_tensor);
            } catch (...) {
                // Some dtypes might not be supported
            }
        }

        // Test with complex tensors if supported
        if (tensor.dtype().isFloatingPoint()) {
            try {
                auto complex_tensor = torch::complex(tensor, torch::zeros_like(tensor));
                auto complex_result = torch::slogdet(complex_tensor);
            } catch (...) {
                // Complex might not be supported in all cases
            }
        }

        // Test edge cases
        if (tensor.numel() > 0) {
            // Test with identity matrix
            if (tensor.dim() >= 2) {
                auto eye_shape = tensor.sizes().vec();
                auto eye_tensor = torch::eye(eye_shape[eye_shape.size()-1], 
                                           torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));
                
                // Expand to match batch dimensions if needed
                if (tensor.dim() > 2) {
                    std::vector<int64_t> expand_shape(tensor.sizes().begin(), tensor.sizes().end()-2);
                    expand_shape.push_back(eye_shape[eye_shape.size()-1]);
                    expand_shape.push_back(eye_shape[eye_shape.size()-1]);
                    eye_tensor = eye_tensor.expand(expand_shape);
                }
                
                auto eye_result = torch::slogdet(eye_tensor);
            }

            // Test with singular matrix (zeros)
            try {
                auto zero_tensor = torch::zeros_like(tensor);
                auto zero_result = torch::slogdet(zero_tensor);
            } catch (...) {
                // Might throw for singular matrices
            }

            // Test with very small values
            try {
                auto small_tensor = tensor * 1e-10;
                auto small_result = torch::slogdet(small_tensor);
            } catch (...) {
                // Might have numerical issues
            }

            // Test with very large values
            try {
                auto large_tensor = tensor * 1e10;
                auto large_result = torch::slogdet(large_tensor);
            } catch (...) {
                // Might overflow
            }
        }

        // Test with different batch sizes
        if (tensor.dim() >= 2) {
            try {
                // Add batch dimension
                auto batched = tensor.unsqueeze(0).repeat({2, 1, 1});
                auto batch_result = torch::slogdet(batched);
            } catch (...) {
                // Might not support batching in some cases
            }
        }

        // Test gradient computation if tensor requires grad
        if (tensor.dtype().isFloatingPoint()) {
            try {
                auto grad_tensor = tensor.clone().detach().requires_grad_(true);
                auto grad_result = torch::slogdet(grad_tensor);
                auto logdet = std::get<1>(grad_result);
                if (logdet.requires_grad()) {
                    auto sum_logdet = logdet.sum();
                    sum_logdet.backward();
                }
            } catch (...) {
                // Gradient computation might fail for singular matrices
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