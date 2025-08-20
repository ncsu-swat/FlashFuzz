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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 2 dimensions for AdaptiveLogSoftmaxWithLoss
        if (input.dim() < 2) {
            input = input.reshape({1, input.numel()});
        }
        
        // Create target tensor (integer labels)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target is 1D and has proper dtype
            if (target.dim() != 1) {
                target = target.reshape({target.numel()});
            }
            
            // Convert to Long type for labels
            target = target.to(torch::kLong);
            
            // Ensure target values are valid indices (non-negative and within range)
            int64_t num_classes = input.size(1);
            target = torch::clamp(target, 0, num_classes - 1);
        } else {
            // Create a default target if we don't have enough data
            target = torch::zeros({input.size(0)}, torch::kLong);
        }
        
        // Ensure batch sizes match
        if (target.size(0) != input.size(0)) {
            target = target.index({torch::indexing::Slice(0, std::min(target.size(0), input.size(0)))});
            if (target.size(0) < input.size(0)) {
                auto padding = torch::zeros({input.size(0) - target.size(0)}, torch::kLong);
                target = torch::cat({target, padding}, 0);
            }
        }
        
        // Parse cutoffs from the remaining data
        std::vector<int64_t> cutoffs;
        int num_cutoffs = 3; // Default number of cutoffs
        
        if (offset + sizeof(int64_t) <= Size) {
            // Extract number of cutoffs from data
            int64_t raw_num_cutoffs;
            std::memcpy(&raw_num_cutoffs, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure reasonable number of cutoffs
            num_cutoffs = std::abs(raw_num_cutoffs) % 10 + 2; // Between 2 and 11 cutoffs
        }
        
        // Generate cutoffs
        int64_t num_classes = input.size(1);
        int64_t in_features = input.size(1);
        cutoffs.push_back(num_classes / 2);  // First cutoff
        
        for (int i = 1; i < num_cutoffs; i++) {
            int64_t cutoff = cutoffs.back() + (num_classes - cutoffs.back()) / 2;
            if (cutoff >= num_classes) break;
            if (cutoff <= cutoffs.back()) break;
            cutoffs.push_back(cutoff);
        }
        
        // Ensure the last cutoff is num_classes
        if (cutoffs.back() < num_classes - 1) {
            cutoffs.push_back(num_classes - 1);
        }
        
        // Parse div_value
        double div_value = 4.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&div_value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure div_value is reasonable
            if (std::isnan(div_value) || std::isinf(div_value)) {
                div_value = 4.0;
            } else {
                div_value = std::abs(div_value);
                if (div_value < 1.0) div_value = 1.0;
                if (div_value > 10.0) div_value = 10.0;
            }
        }
        
        // Parse head_bias
        bool head_bias = false;
        if (offset < Size) {
            head_bias = Data[offset++] & 0x1;
        }
        
        // Create AdaptiveLogSoftmaxWithLoss module
        auto adaptive_log_softmax = torch::nn::AdaptiveLogSoftmaxWithLoss(
            torch::nn::AdaptiveLogSoftmaxWithLossOptions(in_features, num_classes, cutoffs)
                .div_value(div_value)
                .head_bias(head_bias)
        );
        
        // Apply the module
        auto result = adaptive_log_softmax->forward(input, target);
        
        // Access the output and loss
        auto output = std::get<0>(result);
        auto loss = std::get<1>(result);
        
        // Perform some operations on the results to ensure they're used
        auto sum_output = output.sum();
        auto mean_loss = loss.mean();
        
        // Test the predict method
        auto prediction = adaptive_log_softmax->predict(input);
        auto max_pred = prediction.max();
        
        // Test the log_prob method
        auto log_prob = adaptive_log_softmax->log_prob(input);
        auto sum_log_prob = log_prob.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}