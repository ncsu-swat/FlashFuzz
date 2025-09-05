#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: 1 for dtype, 1 for rank, 1 for operation variant
        if (Size < 3) {
            return 0;
        }

        // Create primary tensor from fuzzer input
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get an operation variant byte if available
        uint8_t op_variant = 0;
        if (offset < Size) {
            op_variant = Data[offset++];
        }
        
        // Test signbit operation
        torch::Tensor result;
        
        // Primary operation - torch::signbit
        result = torch::signbit(input_tensor);
        
        // Verify result properties
        if (result.dtype() != torch::kBool) {
            std::cerr << "Unexpected: signbit result is not bool type" << std::endl;
        }
        if (result.sizes() != input_tensor.sizes()) {
            std::cerr << "Unexpected: signbit result shape mismatch" << std::endl;
        }
        
        // Additional operations based on variant to increase coverage
        switch (op_variant % 8) {
            case 0:
                // Test with explicit output tensor
                {
                    torch::Tensor out = torch::empty_like(result);
                    torch::signbit_out(out, input_tensor);
                }
                break;
            case 1:
                // Test with non-contiguous tensor (transpose)
                if (input_tensor.dim() >= 2) {
                    torch::Tensor transposed = input_tensor.transpose(0, 1);
                    torch::Tensor trans_result = torch::signbit(transposed);
                }
                break;
            case 2:
                // Test with view/reshape
                if (input_tensor.numel() > 0) {
                    torch::Tensor flat = input_tensor.flatten();
                    torch::Tensor flat_result = torch::signbit(flat);
                }
                break;
            case 3:
                // Test with slice
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 1) {
                    torch::Tensor sliced = input_tensor.narrow(0, 0, 1);
                    torch::Tensor slice_result = torch::signbit(sliced);
                }
                break;
            case 4:
                // Test with special values injection (if floating point)
                if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                    torch::Tensor special_tensor = input_tensor.clone();
                    // Inject some special values
                    if (input_tensor.numel() >= 4) {
                        special_tensor.view(-1)[0] = std::numeric_limits<float>::infinity();
                        special_tensor.view(-1)[1] = -std::numeric_limits<float>::infinity();
                        special_tensor.view(-1)[2] = std::numeric_limits<float>::quiet_NaN();
                        special_tensor.view(-1)[3] = -0.0f;
                    }
                    torch::Tensor special_result = torch::signbit(special_tensor);
                }
                break;
            case 5:
                // Test with different memory layout (if multidimensional)
                if (input_tensor.dim() >= 2) {
                    torch::Tensor permuted = input_tensor.permute({1, 0});
                    torch::Tensor perm_result = torch::signbit(permuted);
                }
                break;
            case 6:
                // Test with expand/broadcast
                if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
                    std::vector<int64_t> new_shape = input_tensor.sizes().vec();
                    new_shape[0] = 1;  // Make first dimension 1 for broadcasting
                    torch::Tensor reshaped = input_tensor.reshape(new_shape);
                    torch::Tensor expanded = reshaped.expand({3, -1});
                    torch::Tensor exp_result = torch::signbit(expanded);
                }
                break;
            case 7:
                // Test chained operations
                {
                    torch::Tensor res1 = torch::signbit(input_tensor);
                    // Can apply logical operations on bool result
                    torch::Tensor res2 = torch::logical_not(res1);
                    torch::Tensor res3 = torch::logical_and(res1, res2);
                }
                break;
        }
        
        // Additional edge case testing based on remaining bytes
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            switch (edge_case % 5) {
                case 0:
                    // Test empty tensor
                    {
                        torch::Tensor empty = torch::empty({0, 3}, input_tensor.options());
                        torch::Tensor empty_result = torch::signbit(empty);
                    }
                    break;
                case 1:
                    // Test scalar tensor
                    {
                        torch::Tensor scalar = torch::tensor(-3.14f);
                        torch::Tensor scalar_result = torch::signbit(scalar);
                    }
                    break;
                case 2:
                    // Test with different dtype conversion
                    if (input_tensor.dtype() != torch::kDouble) {
                        torch::Tensor converted = input_tensor.to(torch::kDouble);
                        torch::Tensor conv_result = torch::signbit(converted);
                    }
                    break;
                case 3:
                    // Test with contiguous conversion
                    if (!input_tensor.is_contiguous()) {
                        torch::Tensor cont = input_tensor.contiguous();
                        torch::Tensor cont_result = torch::signbit(cont);
                    }
                    break;
                case 4:
                    // Test with requires_grad (if floating point)
                    if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                        torch::Tensor grad_tensor = input_tensor.detach().requires_grad_(true);
                        torch::Tensor grad_result = torch::signbit(grad_tensor);
                        // Note: signbit doesn't support backward, but we test the forward pass
                    }
                    break;
            }
        }
        
        // Test batch processing if we have more data
        if (offset + 10 < Size) {
            // Create a batch of tensors
            std::vector<torch::Tensor> batch;
            size_t batch_size = Data[offset++] % 5 + 1;
            
            for (size_t i = 0; i < batch_size && offset < Size; ++i) {
                try {
                    torch::Tensor batch_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    batch.push_back(batch_tensor);
                    torch::Tensor batch_result = torch::signbit(batch_tensor);
                } catch (const std::exception& e) {
                    // Continue with partial batch
                    break;
                }
            }
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are expected for invalid operations
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}