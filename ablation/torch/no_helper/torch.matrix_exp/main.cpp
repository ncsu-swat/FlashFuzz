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

        // Extract parameters from fuzzer input
        uint8_t rank = Data[offset++] % 5; // Limit rank to 0-4 for memory constraints
        uint8_t dtype_selector = Data[offset++] % 4;
        uint8_t device_selector = Data[offset++] % 2;
        uint8_t requires_grad = Data[offset++] % 2;
        uint8_t layout_selector = Data[offset++] % 2;
        
        // Determine dtype
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kComplexFloat; break;
            case 3: dtype = torch::kComplexDouble; break;
            default: dtype = torch::kFloat32;
        }

        // Determine device
        torch::Device device = device_selector ? torch::kCUDA : torch::kCPU;
        if (device == torch::kCUDA && !torch::cuda::is_available()) {
            device = torch::kCPU;
        }

        // Build tensor shape - matrix_exp expects square matrices
        std::vector<int64_t> shape;
        if (rank == 0) {
            // Scalar case - will likely fail but tests edge case
            shape = {};
        } else if (rank == 1) {
            // 1D tensor - will likely fail but tests edge case
            uint8_t dim = Data[offset++] % 16 + 1;
            shape = {dim};
        } else if (rank == 2) {
            // 2D square matrix - expected case
            uint8_t dim = Data[offset++] % 10 + 1; // Keep matrices small
            shape = {dim, dim};
        } else {
            // Higher dimensional with last two dims as square matrix
            for (int i = 0; i < rank - 2 && offset < Size; i++) {
                shape.push_back((Data[offset++] % 3) + 1);
            }
            uint8_t matrix_dim = (offset < Size) ? (Data[offset++] % 8 + 1) : 3;
            shape.push_back(matrix_dim);
            shape.push_back(matrix_dim);
        }

        // Calculate total elements
        int64_t numel = 1;
        for (auto dim : shape) {
            numel *= dim;
        }
        
        // Limit total size to prevent OOM
        if (numel > 10000) {
            return 0;
        }

        // Create tensor with various initialization methods
        torch::Tensor tensor;
        uint8_t init_method = (offset < Size) ? Data[offset++] % 6 : 0;
        
        torch::TensorOptions options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad(requires_grad);

        if (shape.empty()) {
            // Scalar tensor
            float value = (offset < Size) ? static_cast<float>(Data[offset++]) / 128.0f : 1.0f;
            tensor = torch::tensor(value, options);
        } else {
            switch (init_method) {
                case 0:
                    tensor = torch::zeros(shape, options);
                    break;
                case 1:
                    tensor = torch::ones(shape, options);
                    break;
                case 2:
                    tensor = torch::randn(shape, options);
                    break;
                case 3:
                    tensor = torch::eye(shape.size() >= 2 ? shape[shape.size()-2] : 1, options);
                    if (shape.size() > 2) {
                        // Broadcast eye to batch dimensions
                        std::vector<int64_t> broadcast_shape = shape;
                        tensor = tensor.expand(broadcast_shape);
                    }
                    break;
                case 4: {
                    // Create from raw data if enough bytes available
                    tensor = torch::empty(shape, options);
                    if (dtype == torch::kFloat32 || dtype == torch::kComplexFloat) {
                        float* data_ptr = tensor.data_ptr<float>();
                        int64_t float_numel = (dtype == torch::kComplexFloat) ? numel * 2 : numel;
                        for (int64_t i = 0; i < float_numel && offset < Size; i++) {
                            data_ptr[i] = static_cast<float>(Data[offset++]) / 128.0f;
                        }
                    } else if (dtype == torch::kFloat64 || dtype == torch::kComplexDouble) {
                        double* data_ptr = tensor.data_ptr<double>();
                        int64_t double_numel = (dtype == torch::kComplexDouble) ? numel * 2 : numel;
                        for (int64_t i = 0; i < double_numel && offset < Size; i++) {
                            data_ptr[i] = static_cast<double>(Data[offset++]) / 128.0;
                        }
                    }
                    break;
                }
                case 5: {
                    // Create with special values (inf, nan)
                    tensor = torch::empty(shape, options);
                    if (offset < Size) {
                        uint8_t special_val = Data[offset++] % 4;
                        switch (special_val) {
                            case 0: tensor.fill_(std::numeric_limits<float>::infinity()); break;
                            case 1: tensor.fill_(-std::numeric_limits<float>::infinity()); break;
                            case 2: tensor.fill_(std::numeric_limits<float>::quiet_NaN()); break;
                            case 3: tensor.fill_(0.0); break;
                        }
                    }
                    break;
                }
                default:
                    tensor = torch::randn(shape, options);
            }
        }

        // Apply additional transformations
        if (offset < Size) {
            uint8_t transform = Data[offset++] % 4;
            switch (transform) {
                case 0:
                    // Make tensor contiguous
                    tensor = tensor.contiguous();
                    break;
                case 1:
                    // Transpose if 2D or higher
                    if (tensor.dim() >= 2) {
                        tensor = tensor.transpose(-2, -1);
                    }
                    break;
                case 2:
                    // Add small noise
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        tensor = tensor + torch::randn_like(tensor) * 0.01;
                    }
                    break;
                case 3:
                    // Make symmetric (for square matrices)
                    if (tensor.dim() >= 2 && tensor.size(-1) == tensor.size(-2)) {
                        tensor = (tensor + tensor.transpose(-2, -1)) / 2.0;
                    }
                    break;
            }
        }

        // Call matrix_exp
        torch::Tensor result = torch::matrix_exp(tensor);

        // Perform basic operations on result to ensure it's valid
        if (result.defined()) {
            // Access some properties to trigger potential issues
            auto result_shape = result.sizes();
            auto result_dtype = result.dtype();
            auto result_device = result.device();
            
            // Try to compute some values
            if (result.numel() > 0) {
                auto sum_val = result.sum();
                auto mean_val = result.mean();
                
                // Check for NaN/Inf in result
                auto has_nan = result.isnan().any().item<bool>();
                auto has_inf = result.isinf().any().item<bool>();
            }
        }

    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected for invalid inputs
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}