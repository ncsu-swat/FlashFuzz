#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need sufficient data for meaningful fuzzing
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract parameters from fuzzer input
        uint8_t batch_size = (Data[offset++] % 8) + 1;      // 1-8
        uint8_t max_input_len = (Data[offset++] % 32) + 4;  // 4-35
        uint8_t max_target_len = (Data[offset++] % 16) + 1; // 1-16
        uint8_t num_classes = (Data[offset++] % 20) + 2;    // 2-21 (need at least blank + 1 class)
        int64_t blank = Data[offset++] % num_classes;       // blank label index
        int64_t reduction = Data[offset++] % 3;             // 0=none, 1=mean, 2=sum
        bool zero_infinity = Data[offset++] & 0x01;
        
        // Create log_probs tensor: (T, N, C) - time, batch, classes
        torch::Tensor log_probs = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape and convert log_probs to required format
        int64_t T = max_input_len;
        int64_t N = batch_size;
        int64_t C = num_classes;
        
        log_probs = log_probs.to(torch::kFloat);
        // Ensure we have enough elements, then reshape
        int64_t required_elements = T * N * C;
        if (log_probs.numel() < required_elements) {
            // Expand by repeating
            log_probs = log_probs.flatten();
            while (log_probs.numel() < required_elements) {
                log_probs = torch::cat({log_probs, log_probs});
            }
        }
        log_probs = log_probs.flatten().slice(0, 0, required_elements).reshape({T, N, C});
        
        // Apply log_softmax to get proper log probabilities
        log_probs = torch::log_softmax(log_probs, /*dim=*/2);
        
        // Create input_lengths (N,) - each value should be <= T
        std::vector<int64_t> input_lens_vec(N);
        for (int64_t i = 0; i < N; ++i) {
            if (offset < Size) {
                input_lens_vec[i] = (Data[offset++] % T) + 1;  // 1 to T
            } else {
                input_lens_vec[i] = T;
            }
        }
        torch::Tensor input_lengths = torch::tensor(input_lens_vec, torch::kLong);
        
        // Create target_lengths (N,) - each value should be reasonable
        std::vector<int64_t> target_lens_vec(N);
        int64_t total_target_len = 0;
        for (int64_t i = 0; i < N; ++i) {
            if (offset < Size) {
                // Target length must satisfy: target_len <= input_len (for standard CTC)
                int64_t max_tgt = std::min(static_cast<int64_t>(max_target_len), input_lens_vec[i]);
                target_lens_vec[i] = (Data[offset++] % max_tgt) + 1;  // 1 to max_tgt
            } else {
                target_lens_vec[i] = 1;
            }
            total_target_len += target_lens_vec[i];
        }
        torch::Tensor target_lengths = torch::tensor(target_lens_vec, torch::kLong);
        
        // Create targets as concatenated 1D tensor (sum of target_lengths)
        // Target values should be valid class indices (not the blank label)
        std::vector<int64_t> targets_vec(total_target_len);
        for (int64_t i = 0; i < total_target_len; ++i) {
            if (offset < Size) {
                // Generate target class that is not the blank label
                int64_t val = Data[offset++] % (num_classes - 1);
                if (val >= blank) {
                    val += 1;  // Skip blank index
                }
                targets_vec[i] = val;
            } else {
                // Default to class 0 if it's not blank, else class 1
                targets_vec[i] = (blank == 0) ? 1 : 0;
            }
        }
        torch::Tensor targets = torch::tensor(targets_vec, torch::kLong);
        
        // Apply ctc_loss
        torch::Tensor loss = torch::ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank,
            reduction,
            zero_infinity
        );
        
        // Use the result to prevent optimization
        if (loss.defined()) {
            volatile float sum = loss.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}