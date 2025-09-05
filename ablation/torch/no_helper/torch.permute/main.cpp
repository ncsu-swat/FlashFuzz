#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Extract number of dimensions (1-10)
        uint8_t ndim = (Data[offset] % 10) + 1;
        offset++;

        // Extract shape for each dimension
        std::vector<int64_t> shape;
        for (size_t i = 0; i < ndim && offset < Size; i++) {
            // Allow dimensions from 0 to 10 to test edge cases
            int64_t dim_size = Data[offset] % 11;
            shape.push_back(dim_size);
            offset++;
        }

        // If we don't have enough data for shape, pad with 1s
        while (shape.size() < ndim) {
            shape.push_back(1);
        }

        // Extract dtype choice
        uint8_t dtype_choice = 0;
        if (offset < Size) {
            dtype_choice = Data[offset] % 8;
            offset++;
        }

        // Create tensor with various dtypes
        torch::Tensor tensor;
        torch::TensorOptions options;
        
        switch (dtype_choice) {
            case 0:
                options = torch::TensorOptions().dtype(torch::kFloat32);
                break;
            case 1:
                options = torch::TensorOptions().dtype(torch::kFloat64);
                break;
            case 2:
                options = torch::TensorOptions().dtype(torch::kInt32);
                break;
            case 3:
                options = torch::TensorOptions().dtype(torch::kInt64);
                break;
            case 4:
                options = torch::TensorOptions().dtype(torch::kInt8);
                break;
            case 5:
                options = torch::TensorOptions().dtype(torch::kUInt8);
                break;
            case 6:
                options = torch::TensorOptions().dtype(torch::kBool);
                break;
            case 7:
                options = torch::TensorOptions().dtype(torch::kFloat16);
                break;
            default:
                options = torch::TensorOptions().dtype(torch::kFloat32);
        }

        // Create tensor with the shape
        try {
            tensor = torch::zeros(shape, options);
        } catch (...) {
            // If tensor creation fails (e.g., too large), try smaller shape
            for (auto& s : shape) {
                s = std::min(s, int64_t(3));
            }
            tensor = torch::zeros(shape, options);
        }

        // Fill tensor with some data if we have bytes left
        if (offset < Size && tensor.numel() > 0) {
            size_t bytes_to_copy = std::min(Size - offset, 
                                           static_cast<size_t>(tensor.numel() * tensor.element_size()));
            if (bytes_to_copy > 0 && tensor.data_ptr()) {
                std::memcpy(tensor.data_ptr(), Data + offset, bytes_to_copy);
                offset += bytes_to_copy;
            }
        }

        // Generate permutation dimensions
        std::vector<int64_t> perm_dims;
        
        // Strategy 1: Use fuzzer data to generate permutation
        if (offset + ndim <= Size) {
            std::vector<bool> used(ndim, false);
            for (size_t i = 0; i < ndim && offset < Size; i++) {
                uint8_t dim_idx = Data[offset] % ndim;
                offset++;
                
                // Find next unused dimension
                int attempts = 0;
                while (used[dim_idx] && attempts < ndim) {
                    dim_idx = (dim_idx + 1) % ndim;
                    attempts++;
                }
                
                if (!used[dim_idx]) {
                    perm_dims.push_back(dim_idx);
                    used[dim_idx] = true;
                }
            }
            
            // Fill in any missing dimensions
            for (int64_t i = 0; i < ndim; i++) {
                if (!used[i]) {
                    perm_dims.push_back(i);
                }
            }
        } else {
            // Strategy 2: Reverse permutation as fallback
            for (int64_t i = ndim - 1; i >= 0; i--) {
                perm_dims.push_back(i);
            }
        }

        // Test various edge cases based on remaining fuzzer data
        if (offset < Size) {
            uint8_t test_case = Data[offset] % 5;
            offset++;
            
            switch (test_case) {
                case 0:
                    // Normal permute
                    break;
                case 1:
                    // Identity permutation
                    perm_dims.clear();
                    for (int64_t i = 0; i < ndim; i++) {
                        perm_dims.push_back(i);
                    }
                    break;
                case 2:
                    // Reverse all dimensions
                    perm_dims.clear();
                    for (int64_t i = ndim - 1; i >= 0; i--) {
                        perm_dims.push_back(i);
                    }
                    break;
                case 3:
                    // Swap first two dimensions if possible
                    if (ndim >= 2) {
                        perm_dims.clear();
                        perm_dims.push_back(1);
                        perm_dims.push_back(0);
                        for (int64_t i = 2; i < ndim; i++) {
                            perm_dims.push_back(i);
                        }
                    }
                    break;
                case 4:
                    // Cyclic shift
                    perm_dims.clear();
                    for (int64_t i = 0; i < ndim; i++) {
                        perm_dims.push_back((i + 1) % ndim);
                    }
                    break;
            }
        }

        // Perform the permute operation
        torch::Tensor result = torch::permute(tensor, perm_dims);

        // Verify result properties
        if (result.defined()) {
            // Access some properties to ensure tensor is valid
            auto result_shape = result.sizes();
            auto result_strides = result.strides();
            auto result_numel = result.numel();
            
            // Perform additional operations to exercise the permuted tensor
            if (result_numel > 0) {
                // Try to access first and last element
                if (result.dim() > 0) {
                    auto flat = result.flatten();
                    if (flat.numel() > 0) {
                        flat[0].item();
                        if (flat.numel() > 1) {
                            flat[flat.numel() - 1].item();
                        }
                    }
                }
                
                // Try contiguous conversion
                auto cont = result.contiguous();
                
                // Try another permutation on the result
                if (offset < Size && result.dim() > 1) {
                    std::vector<int64_t> second_perm;
                    for (int64_t i = result.dim() - 1; i >= 0; i--) {
                        second_perm.push_back(i);
                    }
                    torch::Tensor double_perm = torch::permute(result, second_perm);
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}