#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least a few bytes to create tensors and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have more data for the second tensor
        if (offset >= Size - 2) {
            return 0;
        }
        
        // Create second input tensor with same shape as input1
        torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape input2 to match input1's shape for valid cosine similarity computation
        try {
            input2 = input2.view(input1.sizes());
        } catch (...) {
            // If reshape fails, create a tensor with matching shape
            input2 = torch::randn(input1.sizes());
        }
        
        // Create target tensor - must be 1D with size equal to batch dimension
        // Values should be -1 or 1
        int64_t batch_size = input1.dim() > 0 ? input1.size(0) : 1;
        torch::Tensor target;
        
        if (offset < Size) {
            // Use fuzzer data to determine target values
            target = torch::empty({batch_size}, torch::kFloat);
            auto target_acc = target.accessor<float, 1>();
            for (int64_t i = 0; i < batch_size; i++) {
                // Use fuzzer byte to determine -1 or 1
                uint8_t byte = (offset < Size) ? Data[offset++] : 0;
                target_acc[i] = (byte % 2 == 0) ? 1.0f : -1.0f;
            }
        } else {
            // Create default target with all 1s
            target = torch::ones({batch_size});
        }
        
        // Get margin parameter from the input data
        float margin = 0.0f;
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&margin, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Sanitize margin - must be in valid range [-1, 1] typically
            if (std::isnan(margin) || std::isinf(margin)) {
                margin = 0.0f;
            }
            // Clamp to reasonable range
            margin = std::max(-1.0f, std::min(1.0f, margin));
        }
        
        // Get reduction mode from the input data
        uint8_t reduction_raw = 1; // Default to mean
        if (offset < Size) {
            reduction_raw = Data[offset++] % 3;
        }
        
        // Create the CosineEmbeddingLoss module with options
        torch::nn::CosineEmbeddingLossOptions options;
        options.margin(margin);
        
        if (reduction_raw == 0) {
            options.reduction(torch::kNone);
        } else if (reduction_raw == 1) {
            options.reduction(torch::kMean);
        } else {
            options.reduction(torch::kSum);
        }
        
        torch::nn::CosineEmbeddingLoss loss_fn(options);
        
        // Apply the loss function
        torch::Tensor output = loss_fn->forward(input1, input2, target);
        
        // Ensure the computation is executed
        // Use sum() to handle both scalar and non-scalar outputs (when reduction=None)
        volatile float result = output.sum().item<float>();
        (void)result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}