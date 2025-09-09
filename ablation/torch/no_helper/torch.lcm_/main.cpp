#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to create tensors
        if (Size < 16) {
            return 0;
        }

        // Extract tensor properties
        auto dtype_choice = extract_int(Data, Size, offset) % 3;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kInt32; break;
            case 1: dtype = torch::kInt64; break;
            default: dtype = torch::kInt16; break;
        }

        // Extract tensor shapes
        auto shape1_size = (extract_int(Data, Size, offset) % 4) + 1; // 1-4 dimensions
        auto shape2_size = (extract_int(Data, Size, offset) % 4) + 1; // 1-4 dimensions
        
        std::vector<int64_t> shape1, shape2;
        for (int i = 0; i < shape1_size; i++) {
            shape1.push_back((extract_int(Data, Size, offset) % 10) + 1); // 1-10 size per dim
        }
        for (int i = 0; i < shape2_size; i++) {
            shape2.push_back((extract_int(Data, Size, offset) % 10) + 1); // 1-10 size per dim
        }

        // Create tensors with integer values (LCM only works with integers)
        torch::Tensor tensor1, tensor2;
        
        // Test different tensor creation methods
        auto creation_method = extract_int(Data, Size, offset) % 4;
        switch (creation_method) {
            case 0:
                // Random integers
                tensor1 = torch::randint(1, 100, shape1, torch::TensorOptions().dtype(dtype));
                tensor2 = torch::randint(1, 100, shape2, torch::TensorOptions().dtype(dtype));
                break;
            case 1:
                // Ones
                tensor1 = torch::ones(shape1, torch::TensorOptions().dtype(dtype));
                tensor2 = torch::ones(shape2, torch::TensorOptions().dtype(dtype));
                break;
            case 2:
                // Sequential values
                tensor1 = torch::arange(1, torch::prod(torch::tensor(shape1)).item<int64_t>() + 1, torch::TensorOptions().dtype(dtype)).reshape(shape1);
                tensor2 = torch::arange(1, torch::prod(torch::tensor(shape2)).item<int64_t>() + 1, torch::TensorOptions().dtype(dtype)).reshape(shape2);
                break;
            default:
                // Mixed positive and negative values
                tensor1 = torch::randint(-50, 51, shape1, torch::TensorOptions().dtype(dtype));
                tensor2 = torch::randint(-50, 51, shape2, torch::TensorOptions().dtype(dtype));
                // Avoid zeros for LCM
                tensor1 = torch::where(tensor1 == 0, torch::ones_like(tensor1), tensor1);
                tensor2 = torch::where(tensor2 == 0, torch::ones_like(tensor2), tensor2);
                break;
        }

        // Test different broadcasting scenarios
        auto broadcast_test = extract_int(Data, Size, offset) % 5;
        switch (broadcast_test) {
            case 0:
                // Same shape tensors
                if (shape1 != shape2) {
                    tensor2 = tensor2.expand(shape1);
                }
                break;
            case 1:
                // Scalar with tensor
                tensor2 = torch::tensor(extract_int(Data, Size, offset) % 100 + 1, torch::TensorOptions().dtype(dtype));
                break;
            case 2:
                // Different but broadcastable shapes
                if (shape1.size() > 1) {
                    std::vector<int64_t> new_shape = {1};
                    new_shape.insert(new_shape.end(), shape1.begin() + 1, shape1.end());
                    tensor2 = tensor2.reshape(new_shape);
                }
                break;
            case 3:
                // Single element tensor
                tensor1 = torch::tensor(extract_int(Data, Size, offset) % 100 + 1, torch::TensorOptions().dtype(dtype));
                break;
            default:
                // Keep original shapes
                break;
        }

        // Test edge cases with special values
        auto edge_case = extract_int(Data, Size, offset) % 6;
        switch (edge_case) {
            case 0:
                // Large values
                tensor1 = torch::randint(1000, 10000, tensor1.sizes(), torch::TensorOptions().dtype(dtype));
                tensor2 = torch::randint(1000, 10000, tensor2.sizes(), torch::TensorOptions().dtype(dtype));
                break;
            case 1:
                // Small values
                tensor1 = torch::randint(1, 10, tensor1.sizes(), torch::TensorOptions().dtype(dtype));
                tensor2 = torch::randint(1, 10, tensor2.sizes(), torch::TensorOptions().dtype(dtype));
                break;
            case 2:
                // Powers of 2
                tensor1 = torch::pow(2, torch::randint(0, 10, tensor1.sizes(), torch::TensorOptions().dtype(torch::kInt32))).to(dtype);
                tensor2 = torch::pow(2, torch::randint(0, 10, tensor2.sizes(), torch::TensorOptions().dtype(torch::kInt32))).to(dtype);
                break;
            case 3:
                // Prime numbers (small ones)
                auto primes = std::vector<int>{2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
                tensor1.fill_(primes[extract_int(Data, Size, offset) % primes.size()]);
                tensor2.fill_(primes[extract_int(Data, Size, offset) % primes.size()]);
                break;
            case 4:
                // Negative values
                tensor1 = -torch::abs(tensor1);
                tensor2 = -torch::abs(tensor2);
                break;
            default:
                // Keep original values
                break;
        }

        // Store original tensor for comparison
        torch::Tensor original_tensor1 = tensor1.clone();

        // Test torch.lcm_ (in-place operation)
        torch::lcm_(tensor1, tensor2);

        // Verify the result is valid
        if (tensor1.numel() > 0) {
            // Check that result has same shape as original tensor1
            if (!tensor1.sizes().equals(original_tensor1.sizes())) {
                std::cout << "Shape mismatch after lcm_" << std::endl;
            }
            
            // Check that result dtype is preserved
            if (tensor1.dtype() != original_tensor1.dtype()) {
                std::cout << "Dtype changed after lcm_" << std::endl;
            }
        }

        // Test with different tensor properties
        auto device_test = extract_int(Data, Size, offset) % 2;
        if (device_test == 1 && torch::cuda::is_available()) {
            // Test CUDA tensors if available
            auto cuda_tensor1 = torch::randint(1, 100, {3, 3}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
            auto cuda_tensor2 = torch::randint(1, 100, {3, 3}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
            torch::lcm_(cuda_tensor1, cuda_tensor2);
        }

        // Test memory layout variations
        auto layout_test = extract_int(Data, Size, offset) % 3;
        if (layout_test == 1 && tensor1.dim() >= 2) {
            // Test with transposed tensor
            auto transposed = tensor1.transpose(0, 1);
            auto other = torch::randint(1, 100, transposed.sizes(), torch::TensorOptions().dtype(dtype));
            torch::lcm_(transposed, other);
        } else if (layout_test == 2 && tensor1.dim() >= 2) {
            // Test with non-contiguous tensor
            auto sliced = tensor1.slice(0, 0, tensor1.size(0), 2);
            if (sliced.numel() > 0) {
                auto other = torch::randint(1, 100, sliced.sizes(), torch::TensorOptions().dtype(dtype));
                torch::lcm_(sliced, other);
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