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
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a tensor with same shape but different values
            input2 = input1.clone();
        }
        
        // Try to make the shapes compatible for logaddexp
        // If shapes don't match, try broadcasting
        if (input1.sizes() != input2.sizes()) {
            // Try to broadcast tensors if possible
            try {
                // Create dummy operation to check if broadcasting works
                auto dummy = input1 + input2;
            } catch (const std::exception&) {
                // If broadcasting fails, reshape the second tensor to match first
                if (input1.numel() == input2.numel()) {
                    input2 = input2.reshape(input1.sizes());
                } else {
                    // If elements don't match, create a new tensor with matching shape
                    input2 = torch::ones_like(input1);
                }
            }
        }
        
        // Convert tensors to compatible types if needed
        if (input1.scalar_type() != input2.scalar_type()) {
            // Try to promote to a common type
            if (input1.is_floating_point() || input2.is_floating_point()) {
                // Promote to float if either is floating point
                if (!input1.is_floating_point()) {
                    input1 = input1.to(torch::kFloat);
                }
                if (!input2.is_floating_point()) {
                    input2 = input2.to(torch::kFloat);
                }
                
                // Ensure both have the same floating point type
                if (input1.scalar_type() != input2.scalar_type()) {
                    // Promote to higher precision
                    if (input1.scalar_type() == torch::kDouble || input2.scalar_type() == torch::kDouble) {
                        input1 = input1.to(torch::kDouble);
                        input2 = input2.to(torch::kDouble);
                    } else {
                        input1 = input1.to(torch::kFloat);
                        input2 = input2.to(torch::kFloat);
                    }
                }
            } else {
                // For integer types, convert to the same type
                input2 = input2.to(input1.scalar_type());
            }
        }
        
        // Apply logaddexp operation
        torch::Tensor result = torch::logaddexp(input1, input2);
        
        // Try some edge cases with modified inputs
        if (Size > 0 && offset < Size) {
            // Create a tensor with extreme values
            auto extreme_tensor = input1.clone();
            
            // Set some values to extreme values based on the next byte in data
            if (offset < Size) {
                uint8_t selector = Data[offset++];
                
                if (selector % 4 == 0) {
                    // Set to very large positive values
                    extreme_tensor.fill_(1e38);
                } else if (selector % 4 == 1) {
                    // Set to very large negative values
                    extreme_tensor.fill_(-1e38);
                } else if (selector % 4 == 2) {
                    // Set to infinity
                    extreme_tensor.fill_(std::numeric_limits<float>::infinity());
                } else {
                    // Set to NaN
                    extreme_tensor.fill_(std::numeric_limits<float>::quiet_NaN());
                }
                
                // Try logaddexp with extreme values
                torch::Tensor extreme_result = torch::logaddexp(extreme_tensor, input2);
            }
        }
        
        // Try with zero-sized dimensions if we have more data
        if (Size > 0 && offset < Size) {
            uint8_t selector = Data[offset++];
            
            if (selector % 3 == 0) {
                // Create tensors with zero-sized dimensions
                auto zero_dim_tensor1 = torch::empty({0, 2}, input1.options());
                auto zero_dim_tensor2 = torch::empty({0, 2}, input2.options());
                
                // Apply logaddexp to zero-sized tensors
                torch::Tensor zero_result = torch::logaddexp(zero_dim_tensor1, zero_dim_tensor2);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}