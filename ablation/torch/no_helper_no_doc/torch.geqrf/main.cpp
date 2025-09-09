#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions and properties
        auto shape_info = extract_tensor_shape(Data, Size, offset, 2, 4); // 2D tensor, max 4 dimensions
        if (shape_info.empty()) {
            return 0;
        }

        // Extract dtype
        auto dtype = extract_dtype(Data, Size, offset);
        
        // Only test with floating point types as geqrf requires them
        if (dtype != torch::kFloat32 && dtype != torch::kFloat64 && 
            dtype != torch::kComplexFloat && dtype != torch::kComplexDouble) {
            dtype = torch::kFloat32; // Default to float32
        }

        // Extract device
        auto device = extract_device(Data, Size, offset);

        // Create input tensor - geqrf works on matrices (2D tensors)
        torch::Tensor input;
        if (shape_info.size() >= 2) {
            // Ensure we have at least a 2D tensor
            std::vector<int64_t> dims = {shape_info[0], shape_info[1]};
            
            // Limit dimensions to reasonable size to avoid memory issues
            for (auto& dim : dims) {
                dim = std::max(1L, std::min(dim, 100L));
            }
            
            input = create_tensor(dims, dtype, device);
        } else {
            // Default case - create a small square matrix
            input = create_tensor({3, 3}, dtype, device);
        }

        // Fill tensor with random values
        input = input.uniform_(-10.0, 10.0);

        // Test basic geqrf operation
        auto result = torch::geqrf(input);
        auto Q = std::get<0>(result);
        auto R = std::get<1>(result);

        // Verify output tensors are valid
        if (!Q.defined() || !R.defined()) {
            return -1;
        }

        // Test with different input shapes if we have more data
        if (offset < Size - 8) {
            // Extract additional shape parameters
            uint32_t rows = extract_int_in_range(Data, Size, offset, 1, 50);
            uint32_t cols = extract_int_in_range(Data, Size, offset, 1, 50);
            
            torch::Tensor input2 = create_tensor({static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, dtype, device);
            input2 = input2.uniform_(-5.0, 5.0);
            
            auto result2 = torch::geqrf(input2);
            auto Q2 = std::get<0>(result2);
            auto R2 = std::get<1>(result2);
            
            // Verify second result
            if (!Q2.defined() || !R2.defined()) {
                return -1;
            }
        }

        // Test edge cases
        if (offset < Size - 4) {
            uint8_t test_case = Data[offset++];
            
            switch (test_case % 4) {
                case 0: {
                    // Test with very small matrix
                    torch::Tensor small_input = create_tensor({1, 1}, dtype, device);
                    small_input.fill_(1.0);
                    auto small_result = torch::geqrf(small_input);
                    break;
                }
                case 1: {
                    // Test with tall matrix (more rows than columns)
                    torch::Tensor tall_input = create_tensor({10, 5}, dtype, device);
                    tall_input = tall_input.uniform_(-1.0, 1.0);
                    auto tall_result = torch::geqrf(tall_input);
                    break;
                }
                case 2: {
                    // Test with wide matrix (more columns than rows)
                    torch::Tensor wide_input = create_tensor({5, 10}, dtype, device);
                    wide_input = wide_input.uniform_(-1.0, 1.0);
                    auto wide_result = torch::geqrf(wide_input);
                    break;
                }
                case 3: {
                    // Test with matrix containing zeros
                    torch::Tensor zero_input = create_tensor({4, 4}, dtype, device);
                    zero_input.zero_();
                    auto zero_result = torch::geqrf(zero_input);
                    break;
                }
            }
        }

        // Test with batch dimensions if we have complex types
        if ((dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) && offset < Size - 4) {
            torch::Tensor complex_input = create_tensor({2, 3, 3}, dtype, device);
            complex_input = complex_input.uniform_(-2.0, 2.0);
            auto complex_result = torch::geqrf(complex_input);
        }

        // Test gradient computation if input requires grad
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            torch::Tensor grad_input = create_tensor({4, 4}, dtype, device);
            grad_input.requires_grad_(true);
            grad_input = grad_input.uniform_(-1.0, 1.0);
            
            auto grad_result = torch::geqrf(grad_input);
            auto Q_grad = std::get<0>(grad_result);
            auto R_grad = std::get<1>(grad_result);
            
            // Create a simple loss and backpropagate
            auto loss = Q_grad.sum() + R_grad.sum();
            loss.backward();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}