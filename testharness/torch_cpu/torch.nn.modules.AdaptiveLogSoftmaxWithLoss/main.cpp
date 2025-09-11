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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (logits)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (labels)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target has integer type for labels
            if (target.scalar_type() != torch::kLong) {
                target = target.to(torch::kLong);
            }
            
            // Ensure target values are non-negative
            target = torch::abs(target);
        } else {
            // Create a default target if we don't have enough data
            if (input.dim() > 0) {
                auto batch_size = input.size(0);
                target = torch::randint(0, std::max<int64_t>(1, input.size(1) - 1), {batch_size}, torch::kLong);
            } else {
                target = torch::zeros({1}, torch::kLong);
            }
        }
        
        // Extract parameters for AdaptiveLogSoftmaxWithLoss
        int64_t in_features = 0;
        int64_t n_classes = 0;
        if (input.dim() > 0) {
            if (input.dim() > 1) {
                in_features = input.size(1);
                n_classes = input.size(1);
            } else {
                in_features = input.size(0);
                n_classes = input.size(0);
            }
        } else {
            in_features = 10; // Default value if input is a scalar
            n_classes = 10; // Default value if input is a scalar
        }
        
        // Ensure n_classes is at least 2
        n_classes = std::max<int64_t>(2, n_classes);
        in_features = std::max<int64_t>(2, in_features);
        
        // Extract cutoffs from the remaining data
        std::vector<int64_t> cutoffs;
        if (offset + 1 < Size) {
            uint8_t num_cutoffs = Data[offset++] % 5; // Limit number of cutoffs
            
            for (uint8_t i = 0; i < num_cutoffs && offset + sizeof(int64_t) <= Size; i++) {
                int64_t cutoff_value;
                std::memcpy(&cutoff_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure cutoff is positive and less than n_classes
                cutoff_value = std::abs(cutoff_value) % (n_classes - 1) + 1;
                
                // Add cutoff if it's not already in the vector and is valid
                if (std::find(cutoffs.begin(), cutoffs.end(), cutoff_value) == cutoffs.end()) {
                    cutoffs.push_back(cutoff_value);
                }
            }
            
            // Sort cutoffs in ascending order
            std::sort(cutoffs.begin(), cutoffs.end());
        }
        
        // If no valid cutoffs, add some default ones
        if (cutoffs.empty()) {
            int64_t cutoff1 = n_classes / 4;
            int64_t cutoff2 = n_classes / 2;
            
            if (cutoff1 > 0) cutoffs.push_back(cutoff1);
            if (cutoff2 > cutoff1 && cutoff2 < n_classes) cutoffs.push_back(cutoff2);
        }
        
        // Get div_value
        double div_value = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&div_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure div_value is positive and reasonable
            div_value = std::abs(div_value);
            if (std::isnan(div_value) || std::isinf(div_value) || div_value < 1e-6) {
                div_value = 1.0;
            }
            div_value = std::min(div_value, 10.0); // Cap at a reasonable value
        }
        
        // Get head_bias flag
        bool head_bias = false;
        if (offset < Size) {
            head_bias = Data[offset++] & 0x1;
        }
        
        // Create the AdaptiveLogSoftmaxWithLoss module
        torch::nn::AdaptiveLogSoftmaxWithLoss module(
            torch::nn::AdaptiveLogSoftmaxWithLossOptions(in_features, n_classes, cutoffs)
                .div_value(div_value)
                .head_bias(head_bias)
        );
        
        // Reshape input if needed to match expected format
        if (input.dim() == 0) {
            input = input.reshape({1, 1});
        } else if (input.dim() == 1) {
            input = input.reshape({1, input.size(0)});
        }
        
        // Reshape target if needed
        if (target.dim() == 0) {
            target = target.reshape({1});
        }
        
        // Ensure target has the right shape (batch_size)
        if (target.size(0) != input.size(0)) {
            target = target.reshape({input.size(0)});
        }
        
        // Ensure target values are within valid range
        target = target % n_classes;
        
        // Forward pass
        auto output = module->forward(input, target);
        
        // Get output and loss
        auto output_tensor = std::get<0>(output);
        auto loss = std::get<1>(output);
        
        // Test log_prob method
        auto log_prob = module->log_prob(input);
        
        // Test predict method
        auto predict = module->predict(input);
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
