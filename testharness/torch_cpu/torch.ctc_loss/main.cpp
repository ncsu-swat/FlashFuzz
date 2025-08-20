#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 10) {
            return 0;
        }
        
        // Create log probs tensor (BxTxN)
        torch::Tensor log_probs = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create targets tensor (BxS)
        torch::Tensor targets;
        if (offset < Size) {
            targets = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create input_lengths tensor (B)
        torch::Tensor input_lengths;
        if (offset < Size) {
            input_lengths = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Create target_lengths tensor (B)
        torch::Tensor target_lengths;
        if (offset < Size) {
            target_lengths = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Get reduction type from input data
        int64_t reduction = 0;
        if (offset < Size) {
            reduction = static_cast<int64_t>(Data[offset++]) % 3;
        }
        
        // Get zero_infinity flag from input data
        bool zero_infinity = false;
        if (offset < Size) {
            zero_infinity = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Convert tensors to appropriate types if needed
        if (log_probs.scalar_type() != torch::kFloat && 
            log_probs.scalar_type() != torch::kDouble) {
            log_probs = log_probs.to(torch::kFloat);
        }
        
        if (targets.scalar_type() != torch::kInt && 
            targets.scalar_type() != torch::kLong) {
            targets = targets.to(torch::kLong);
        }
        
        if (input_lengths.scalar_type() != torch::kInt && 
            input_lengths.scalar_type() != torch::kLong) {
            input_lengths = input_lengths.to(torch::kLong);
        }
        
        if (target_lengths.scalar_type() != torch::kInt && 
            target_lengths.scalar_type() != torch::kLong) {
            target_lengths = target_lengths.to(torch::kLong);
        }
        
        // Make input_lengths and target_lengths 1D if they're not
        if (input_lengths.dim() > 1) {
            input_lengths = input_lengths.flatten();
        }
        
        if (target_lengths.dim() > 1) {
            target_lengths = target_lengths.flatten();
        }
        
        // Apply ctc_loss
        torch::Tensor loss = torch::ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction,
            zero_infinity
        );
        
        // Ensure we use the result to prevent optimization from removing the computation
        if (loss.defined()) {
            volatile float sum = loss.sum().item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}