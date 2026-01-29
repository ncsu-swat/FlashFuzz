#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        // Early return if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (logits)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors
        if (input.numel() == 0) {
            return 0;
        }
        
        // Make input float and require gradients
        input = input.to(torch::kFloat32).requires_grad_(true);
        
        // Create target tensor (same shape as input)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            target = torch::rand_like(input);
        }
        
        // Ensure target has same shape as input
        if (input.sizes() != target.sizes()) {
            target = torch::rand_like(input);
        }
        
        // Ensure target values are between 0 and 1 for BCE
        target = target.to(torch::kFloat32);
        target = torch::sigmoid(target); // Maps to (0, 1) range
        
        // Extract options for BCEWithLogitsLoss from the remaining data
        int reduction_type = 0; // 0=mean, 1=sum, 2=none
        float pos_weight_value = 1.0f;
        
        if (offset < Size) {
            reduction_type = Data[offset++] % 3;
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&pos_weight_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure pos_weight is positive and not extreme
            if (std::isnan(pos_weight_value) || std::isinf(pos_weight_value)) {
                pos_weight_value = 1.0f;
            } else {
                pos_weight_value = std::abs(pos_weight_value) + 0.1f;
                pos_weight_value = std::min(pos_weight_value, 10.0f);
            }
        }
        
        // Determine whether to use pos_weight and weight
        bool use_pos_weight = false;
        bool use_weight = false;
        if (offset < Size) {
            use_pos_weight = (Data[offset] & 0x01) != 0;
            use_weight = (Data[offset] & 0x02) != 0;
            offset++;
        }
        
        // Create BCEWithLogitsLoss with options
        torch::nn::BCEWithLogitsLossOptions options;
        
        // Set reduction
        if (reduction_type == 0) {
            options.reduction(torch::kMean);
        } else if (reduction_type == 1) {
            options.reduction(torch::kSum);
        } else {
            options.reduction(torch::kNone);
        }
        
        // pos_weight should be a 1D tensor broadcastable with target
        // Typically it has size equal to the last dimension (number of classes)
        if (use_pos_weight) {
            int64_t last_dim_size = input.size(-1);
            torch::Tensor pos_weight = torch::ones({last_dim_size}) * pos_weight_value;
            options.pos_weight(pos_weight);
        }
        
        // weight tensor must have the same shape as input/target
        if (use_weight) {
            float weight_value = 1.0f;
            if (offset + sizeof(float) <= Size) {
                std::memcpy(&weight_value, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (std::isnan(weight_value) || std::isinf(weight_value)) {
                    weight_value = 1.0f;
                } else {
                    weight_value = std::abs(weight_value) + 0.1f;
                    weight_value = std::min(weight_value, 10.0f);
                }
            }
            torch::Tensor weight = torch::ones_like(input) * weight_value;
            options.weight(weight);
        }
        
        // Create the loss function
        torch::nn::BCEWithLogitsLoss bce_loss(options);
        
        // Apply the loss function
        torch::Tensor loss;
        try {
            loss = bce_loss(input, target);
        } catch (...) {
            // Shape mismatch or other expected errors
            return 0;
        }
        
        // Verify loss is valid
        if (!loss.defined()) {
            return 0;
        }
        
        // Compute gradients
        try {
            if (reduction_type == 2) {
                // For 'none' reduction, sum before backward
                loss.sum().backward();
            } else {
                loss.backward();
            }
        } catch (...) {
            // Gradient computation can fail for various reasons
        }
        
        // Test forward with different input types
        if (offset < Size) {
            uint8_t test_type = Data[offset++] % 3;
            
            try {
                torch::Tensor test_input;
                torch::Tensor test_target;
                
                if (test_type == 0) {
                    // 1D tensors
                    int64_t len = std::max((int64_t)1, input.numel() % 100 + 1);
                    test_input = torch::randn({len}).requires_grad_(true);
                    test_target = torch::rand({len});
                } else if (test_type == 1) {
                    // 2D tensors (batch, classes)
                    int64_t batch = std::max((int64_t)1, (int64_t)(Data[offset % Size] % 16 + 1));
                    int64_t classes = std::max((int64_t)1, (int64_t)(Data[(offset + 1) % Size] % 32 + 1));
                    test_input = torch::randn({batch, classes}).requires_grad_(true);
                    test_target = torch::rand({batch, classes});
                } else {
                    // 3D tensors
                    int64_t d1 = std::max((int64_t)1, (int64_t)(Data[offset % Size] % 8 + 1));
                    int64_t d2 = std::max((int64_t)1, (int64_t)(Data[(offset + 1) % Size] % 8 + 1));
                    int64_t d3 = std::max((int64_t)1, (int64_t)(Data[(offset + 2) % Size] % 8 + 1));
                    test_input = torch::randn({d1, d2, d3}).requires_grad_(true);
                    test_target = torch::rand({d1, d2, d3});
                }
                
                // Create fresh loss module for this test
                torch::nn::BCEWithLogitsLossOptions test_options;
                test_options.reduction(torch::kMean);
                torch::nn::BCEWithLogitsLoss test_loss(test_options);
                
                torch::Tensor result = test_loss(test_input, test_target);
                result.backward();
            } catch (...) {
                // Ignore errors in variant testing
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