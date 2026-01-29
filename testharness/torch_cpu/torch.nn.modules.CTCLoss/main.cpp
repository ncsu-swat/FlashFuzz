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
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzz data
        uint8_t T_val = (Data[offset++] % 10) + 1;  // input sequence length (1-10)
        uint8_t N_val = (Data[offset++] % 4) + 1;   // batch size (1-4)
        uint8_t C_val = (Data[offset++] % 20) + 2;  // num classes including blank (2-21)
        uint8_t S_val = (Data[offset++] % 8) + 1;   // max target length (1-8)
        
        // Ensure S <= T (target length can't exceed input length for valid CTC)
        if (S_val > T_val) {
            S_val = T_val;
        }

        int T = static_cast<int>(T_val);
        int N = static_cast<int>(N_val);
        int C = static_cast<int>(C_val);
        int S = static_cast<int>(S_val);

        // Get reduction type from input data
        int reduction_type = 1; // default to Mean
        if (offset < Size) {
            reduction_type = Data[offset++] % 3;
        }

        // Get zero_infinity flag from input data
        bool zero_infinity = false;
        if (offset < Size) {
            zero_infinity = Data[offset++] & 0x1;
        }

        // Get blank label (default 0)
        int64_t blank = 0;
        if (offset < Size) {
            blank = Data[offset++] % C;  // blank must be in valid range
        }

        // Create log_probs tensor: (T, N, C)
        // Use remaining data to seed random values
        torch::manual_seed(Size > 0 ? Data[offset % Size] : 0);
        torch::Tensor log_probs = torch::randn({T, N, C}, torch::kFloat);
        // Apply log_softmax to get proper log probabilities
        log_probs = torch::log_softmax(log_probs, /*dim=*/2);
        log_probs.set_requires_grad(true);

        // Create targets tensor: (N, S) with values in [0, C-1] excluding blank
        // For simplicity, use values in range [0, C-1] but avoid blank
        std::vector<int64_t> target_data(N * S);
        for (int i = 0; i < N * S; ++i) {
            int64_t val = 0;
            if (offset < Size) {
                val = Data[offset++] % C;
                // Avoid blank label in targets
                if (val == blank) {
                    val = (val + 1) % C;
                }
            }
            target_data[i] = val;
        }
        torch::Tensor targets = torch::tensor(target_data, torch::kLong).reshape({N, S});

        // Create input_lengths tensor: (N,) all set to T
        std::vector<int64_t> input_len_data(N, T);
        torch::Tensor input_lengths = torch::tensor(input_len_data, torch::kLong);

        // Create target_lengths tensor: (N,) values between 1 and S
        std::vector<int64_t> target_len_data(N);
        for (int i = 0; i < N; ++i) {
            int64_t len = S;
            if (offset < Size) {
                len = (Data[offset++] % S) + 1;  // length between 1 and S
            }
            target_len_data[i] = len;
        }
        torch::Tensor target_lengths = torch::tensor(target_len_data, torch::kLong);

        // Create CTCLoss module with options
        torch::nn::CTCLossOptions options;
        options.blank(blank).zero_infinity(zero_infinity);
        
        // Set reduction using proper enum type
        switch (reduction_type) {
            case 0:
                options.reduction(torch::kNone);
                break;
            case 1:
                options.reduction(torch::kMean);
                break;
            case 2:
                options.reduction(torch::kSum);
                break;
        }
        
        torch::nn::CTCLoss ctc_loss(options);

        // Apply CTCLoss - use inner try-catch for expected shape errors
        torch::Tensor loss;
        try {
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths);
        } catch (const std::exception& e) {
            // Shape mismatches or invalid inputs are expected
            return 0;
        }

        // Try backward pass if possible
        if (loss.defined() && loss.numel() > 0 && loss.requires_grad()) {
            try {
                if (reduction_type == 0) {
                    loss.sum().backward();
                } else {
                    loss.backward();
                }
            } catch (const std::exception& e) {
                // Ignore backward pass errors
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}