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
        // Need enough data to construct meaningful inputs
        if (Size < 20) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzzer data
        uint8_t T_raw = Data[offset++];  // input sequence length
        uint8_t N_raw = Data[offset++];  // batch size
        uint8_t C_raw = Data[offset++];  // number of classes (including blank)
        uint8_t reduction_byte = Data[offset++];
        uint8_t zero_infinity_byte = Data[offset++];
        uint8_t blank_byte = Data[offset++];

        // Constrain dimensions to reasonable values
        int64_t T = (T_raw % 32) + 1;  // 1-32 input length
        int64_t N = (N_raw % 8) + 1;   // 1-8 batch size
        int64_t C = (C_raw % 16) + 2;  // 2-17 classes (need at least 2 for blank + 1 label)
        
        int64_t blank = blank_byte % C;  // blank index must be < C

        // Create log_probs tensor: (T, N, C)
        // Should contain log-probabilities (output of log_softmax)
        torch::Tensor log_probs = torch::randn({T, N, C}, torch::kFloat32);
        log_probs = torch::log_softmax(log_probs, /*dim=*/2);
        log_probs.set_requires_grad(true);

        // Create input_lengths: 1D tensor of size N, each value <= T
        std::vector<int64_t> input_lens_vec(N);
        for (int64_t i = 0; i < N; i++) {
            if (offset < Size) {
                input_lens_vec[i] = (Data[offset++] % T) + 1;  // 1 to T
            } else {
                input_lens_vec[i] = T;
            }
        }
        torch::Tensor input_lengths = torch::tensor(input_lens_vec, torch::kInt64);

        // Create target_lengths: 1D tensor of size N
        // Each target length should be <= corresponding input length (for valid CTC)
        std::vector<int64_t> target_lens_vec(N);
        int64_t total_target_len = 0;
        for (int64_t i = 0; i < N; i++) {
            if (offset < Size) {
                // Target length must be <= input length for valid CTC
                int64_t max_target_len = std::max(int64_t(1), input_lens_vec[i]);
                target_lens_vec[i] = (Data[offset++] % max_target_len) + 1;
            } else {
                target_lens_vec[i] = 1;
            }
            total_target_len += target_lens_vec[i];
        }
        torch::Tensor target_lengths = torch::tensor(target_lens_vec, torch::kInt64);

        // Create targets: 1D tensor of concatenated targets
        // Values should be in [0, C-1] but not equal to blank
        std::vector<int64_t> targets_vec(total_target_len);
        for (int64_t i = 0; i < total_target_len; i++) {
            if (offset < Size) {
                // Generate label that is not blank
                int64_t label = Data[offset++] % (C - 1);
                if (label >= blank) {
                    label++;  // Skip the blank index
                }
                targets_vec[i] = label;
            } else {
                targets_vec[i] = (blank == 0) ? 1 : 0;
            }
        }
        torch::Tensor targets = torch::tensor(targets_vec, torch::kInt64);

        // Determine reduction mode - track with int for later comparison
        int reduction_mode = reduction_byte % 3;  // 0=None, 1=Mean, 2=Sum
        bool zero_infinity = zero_infinity_byte & 0x1;

        // Create CTCLoss module with options
        torch::nn::CTCLossOptions options;
        options.blank(blank).zero_infinity(zero_infinity);
        
        switch (reduction_mode) {
            case 0:
                options.reduction(torch::kNone);
                break;
            case 1:
                options.reduction(torch::kMean);
                break;
            case 2:
            default:
                options.reduction(torch::kSum);
                break;
        }
        
        torch::nn::CTCLoss ctc_loss(options);

        // Apply CTCLoss
        torch::Tensor loss = ctc_loss(log_probs, targets, input_lengths, target_lengths);

        // Force computation
        if (reduction_mode == 0) {  // kNone
            loss.sum().item<float>();
        } else {
            loss.item<float>();
        }

        // Test backward pass to exercise more code paths
        try {
            loss.sum().backward();
        } catch (...) {
            // Backward may fail for some configurations, that's OK
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}