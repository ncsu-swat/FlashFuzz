#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some data to proceed
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor with requires_grad for backward pass testing
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        input = input.to(torch::kFloat32).detach().requires_grad_(true);

        // Create target tensor with same shape as input
        // Target must contain only -1 or 1 values for HingeEmbeddingLoss
        torch::Tensor target = torch::ones_like(input);
        
        // Use fuzzer data to determine which elements are -1
        if (offset < Size) {
            auto flat_target = target.flatten();
            int64_t num_elements = flat_target.numel();
            for (int64_t i = 0; i < num_elements && offset < Size; i++) {
                if (Data[offset % Size] % 2 == 0) {
                    flat_target[i] = -1.0f;
                }
                offset++;
            }
            target = flat_target.view(input.sizes());
        }

        // Extract margin parameter from the input data
        double margin = 1.0;
        if (offset + sizeof(float) <= Size) {
            float margin_raw;
            std::memcpy(&margin_raw, Data + offset, sizeof(float));
            offset += sizeof(float);

            // Ensure margin is a reasonable positive value
            if (!std::isnan(margin_raw) && !std::isinf(margin_raw)) {
                margin = std::abs(static_cast<double>(margin_raw));
                // Clamp to reasonable range
                if (margin > 100.0) margin = 100.0;
            }
        }

        // Extract reduction parameter from the input data
        // Use torch::k* enum values instead of torch::Reduction::Reduction
        int reduction_choice = 1; // Default to Mean
        if (offset < Size) {
            reduction_choice = Data[offset++] % 3;
        }

        // Create HingeEmbeddingLoss module with the extracted parameters
        torch::nn::HingeEmbeddingLossOptions options;
        options.margin(margin);
        
        // Set reduction using the correct enum type
        switch (reduction_choice) {
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
        
        auto loss_fn = torch::nn::HingeEmbeddingLoss(options);

        // Apply the loss function
        torch::Tensor loss;
        try {
            loss = loss_fn->forward(input, target);
        } catch (const c10::Error &e) {
            // Shape mismatches or other expected errors
            return 0;
        }

        // Ensure the computation is executed
        if (loss.numel() > 0) {
            if (loss.numel() == 1) {
                (void)loss.item<float>();
                
                // Try backward pass for scalar loss
                try {
                    loss.backward();
                } catch (...) {
                    // Backward may fail for certain configurations
                }
            } else {
                // For non-reduced output, just access some elements
                (void)loss.sum().item<float>();
            }
        }

        // Also test the functional interface
        try {
            torch::nn::functional::HingeEmbeddingLossFuncOptions func_options;
            func_options.margin(margin);
            switch (reduction_choice) {
                case 0:
                    func_options.reduction(torch::kNone);
                    break;
                case 1:
                    func_options.reduction(torch::kMean);
                    break;
                case 2:
                    func_options.reduction(torch::kSum);
                    break;
            }
            
            auto functional_loss = torch::nn::functional::hinge_embedding_loss(
                input.detach().requires_grad_(true),
                target,
                func_options
            );
            if (functional_loss.numel() == 1) {
                (void)functional_loss.item<float>();
            }
        } catch (...) {
            // Functional interface might fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}