#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (logits)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (labels)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has same shape as input
            if (target.sizes() != input.sizes()) {
                target = target.reshape_as(input);
            }
            
            // For binary cross entropy, target values should be between 0 and 1
            // We'll clamp them to ensure valid values
            target = torch::clamp(target, 0.0, 1.0);
        } else {
            // If we don't have enough data for a second tensor, create one with same shape as input
            target = torch::rand_like(input);
        }
        
        // Create optional weight tensor
        torch::Tensor weight;
        bool use_weight = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (offset < Size) {
                use_weight = true;
                weight = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure weight has same shape as input
                if (weight.sizes() != input.sizes()) {
                    weight = weight.reshape_as(input);
                }
                
                // Weights should be positive
                weight = torch::abs(weight) + 1e-6;
            }
        }
        
        // Create optional pos_weight tensor
        torch::Tensor pos_weight;
        bool use_pos_weight = false;
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (offset < Size) {
                use_pos_weight = true;
                pos_weight = fuzzer_utils::createTensor(Data, Size, offset);
                
                // pos_weight should be a 1D tensor with size equal to the number of classes (C)
                // For simplicity, we'll use a single value
                pos_weight = torch::abs(pos_weight) + 1e-6;
                
                // Reshape to match expected dimensions
                if (input.dim() > 0) {
                    int64_t C = input.size(input.dim() - 1);
                    pos_weight = pos_weight.reshape({C});
                } else {
                    pos_weight = torch::tensor(1.0f);
                }
            }
        }
        
        // Determine reduction method
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        if (offset < Size) {
            uint8_t reduction_byte = Data[offset++];
            if (reduction_byte % 3 == 0) {
                reduction = torch::Reduction::None;
            } else if (reduction_byte % 3 == 1) {
                reduction = torch::Reduction::Mean;
            } else {
                reduction = torch::Reduction::Sum;
            }
        }
        
        // Convert tensors to float if needed for the operation
        if (input.scalar_type() != torch::kFloat && 
            input.scalar_type() != torch::kDouble) {
            input = input.to(torch::kFloat);
        }
        
        if (target.scalar_type() != torch::kFloat && 
            target.scalar_type() != torch::kDouble) {
            target = target.to(input.scalar_type());
        }
        
        if (use_weight && weight.scalar_type() != input.scalar_type()) {
            weight = weight.to(input.scalar_type());
        }
        
        if (use_pos_weight && pos_weight.scalar_type() != input.scalar_type()) {
            pos_weight = pos_weight.to(input.scalar_type());
        }
        
        // Apply binary_cross_entropy_with_logits with different combinations of optional parameters
        torch::Tensor output;
        
        if (use_weight && use_pos_weight) {
            output = torch::binary_cross_entropy_with_logits(
                input, target, weight, pos_weight, reduction);
        } else if (use_weight) {
            output = torch::binary_cross_entropy_with_logits(
                input, target, weight, {}, reduction);
        } else if (use_pos_weight) {
            output = torch::binary_cross_entropy_with_logits(
                input, target, {}, pos_weight, reduction);
        } else {
            output = torch::binary_cross_entropy_with_logits(
                input, target, {}, {}, reduction);
        }
        
        // Ensure the output is valid
        if (output.isnan().any().item<bool>() || 
            output.isinf().any().item<bool>()) {
            return 0;  // Discard inputs that produce NaN or Inf
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
