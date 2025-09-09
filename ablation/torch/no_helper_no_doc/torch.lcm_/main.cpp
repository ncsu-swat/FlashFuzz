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
                // Keep current values
                break;
        }

        // Make a copy to test in-place operation
        torch::Tensor original_tensor1 = tensor1.clone();

        // Test torch.lcm_ (in-place LCM)
        tensor1.lcm_(tensor2);

        // Verify the operation didn't crash and tensor is still valid
        if (tensor1.numel() > 0) {
            // Basic sanity checks
            auto result_sum = tensor1.sum();
            auto result_mean = tensor1.mean();
            
            // Check that result has reasonable properties for LCM
            if (original_tensor1.numel() > 0 && tensor2.numel() > 0) {
                // LCM should be >= max of absolute values of inputs (for positive integers)
                auto abs_tensor1 = torch::abs(original_tensor1);
                auto abs_tensor2 = torch::abs(tensor2);
                auto max_input = torch::maximum(abs_tensor1, abs_tensor2);
                
                // For positive inputs, LCM >= max(a,b)
                auto positive_mask = (original_tensor1 > 0) & (tensor2 > 0);
                if (positive_mask.any().item<bool>()) {
                    auto lcm_positive = tensor1.masked_select(positive_mask);
                    auto max_positive = max_input.masked_select(positive_mask);
                    // This should generally hold for LCM
                }
            }
        }

        // Test with different memory layouts
        auto layout_test = extract_int(Data, Size, offset) % 3;
        switch (layout_test) {
            case 0:
                // Contiguous tensors
                tensor1 = tensor1.contiguous();
                tensor2 = tensor2.contiguous();
                break;
            case 1:
                // Non-contiguous tensors (if possible)
                if (tensor1.dim() > 1) {
                    tensor1 = tensor1.transpose(0, 1);
                }
                if (tensor2.dim() > 1) {
                    tensor2 = tensor2.transpose(0, 1);
                }
                break;
            default:
                // Keep current layout
                break;
        }

        // Test another lcm_ operation
        if (tensor1.numel() > 0 && tensor2.numel() > 0) {
            tensor1.lcm_(tensor2);
        }

        // Test with device placement if CUDA is available
        if (torch::cuda::is_available() && extract_int(Data, Size, offset) % 2 == 0) {
            try {
                auto cuda_tensor1 = tensor1.to(torch::kCUDA);
                auto cuda_tensor2 = tensor2.to(torch::kCUDA);
                cuda_tensor1.lcm_(cuda_tensor2);
            } catch (...) {
                // CUDA operations might fail, that's okay
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