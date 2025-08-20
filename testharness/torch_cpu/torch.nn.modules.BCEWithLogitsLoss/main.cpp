#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Early return if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor (logits)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create target tensor (same shape as input)
        torch::Tensor target;
        if (offset < Size) {
            target = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure target values are between 0 and 1 for BCE
            target = torch::clamp(target, 0.0, 1.0);
        } else {
            // If we don't have enough data, create a matching target
            target = torch::rand_like(input);
        }
        
        // Try to make shapes compatible if they differ
        if (input.sizes() != target.sizes()) {
            // Attempt to broadcast or reshape
            if (input.dim() > 0 && target.dim() > 0) {
                // Try to broadcast to the same shape
                try {
                    // Get the first element and broadcast it
                    if (target.numel() > 0) {
                        auto first_elem = target.flatten()[0];
                        target = torch::ones_like(input) * first_elem;
                    } else {
                        target = torch::zeros_like(input);
                    }
                } catch (...) {
                    // If broadcasting fails, create a new compatible target
                    target = torch::rand_like(input);
                }
            } else {
                // Handle scalar case or empty tensor
                target = torch::rand_like(input);
            }
        }
        
        // Extract options for BCEWithLogitsLoss from the remaining data
        bool reduction_mean = true;
        float pos_weight_value = 1.0f;
        
        if (offset + 1 < Size) {
            // Use a byte to determine reduction
            uint8_t reduction_byte = Data[offset++];
            reduction_mean = (reduction_byte % 3 != 0); // 2/3 chance of using mean reduction
        }
        
        if (offset + sizeof(float) <= Size) {
            // Extract pos_weight value
            std::memcpy(&pos_weight_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure pos_weight is positive and not too extreme
            if (std::isnan(pos_weight_value) || std::isinf(pos_weight_value)) {
                pos_weight_value = 1.0f;
            } else {
                pos_weight_value = std::abs(pos_weight_value) + 0.1f; // Ensure positive
                pos_weight_value = std::min(pos_weight_value, 10.0f); // Cap at reasonable value
            }
        }
        
        // Create pos_weight tensor (same shape as first dimension of input)
        torch::Tensor pos_weight;
        if (input.dim() > 0 && input.size(0) > 0) {
            pos_weight = torch::ones(input.size(0)) * pos_weight_value;
        } else {
            pos_weight = torch::tensor(pos_weight_value);
        }
        
        // Create BCEWithLogitsLoss with different configurations
        torch::nn::BCEWithLogitsLossOptions options;
        
        // Set reduction
        if (reduction_mean) {
            options.reduction(torch::kMean);
        } else {
            options.reduction(torch::kSum);
        }
        
        // Try with and without pos_weight
        bool use_pos_weight = false;
        if (offset < Size) {
            use_pos_weight = (Data[offset++] % 2 == 0);
        }
        
        if (use_pos_weight) {
            options.pos_weight(pos_weight);
        }
        
        // Create the loss function
        torch::nn::BCEWithLogitsLoss bce_loss(options);
        
        // Apply the loss function
        torch::Tensor loss = bce_loss(input, target);
        
        // Ensure loss is valid
        if (loss.defined() && !loss.isnan().any().item<bool>() && !loss.isinf().any().item<bool>()) {
            // Compute gradients if possible
            if (input.requires_grad()) {
                loss.backward();
            }
        }
        
        // Try with weight tensor
        if (offset < Size && input.dim() > 0) {
            try {
                torch::Tensor weight;
                if (offset < Size) {
                    weight = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Ensure weight is positive
                    weight = torch::abs(weight) + 0.1;
                    
                    // Try to make weight compatible with input
                    if (input.dim() > 1 && weight.dim() > 0) {
                        // Reshape weight to match class dimension
                        if (weight.numel() != input.size(1) && weight.numel() > 0) {
                            weight = weight.view({-1}).index({torch::indexing::Slice(0, std::min(weight.numel(), input.size(1)))});
                            if (weight.numel() < input.size(1)) {
                                weight = weight.repeat(input.size(1) / weight.numel() + 1).index({torch::indexing::Slice(0, input.size(1))});
                            }
                        }
                    }
                    
                    // Create new options with weight
                    torch::nn::BCEWithLogitsLossOptions weighted_options;
                    weighted_options.reduction(torch::kMean).weight(weight);
                    
                    if (use_pos_weight) {
                        weighted_options.pos_weight(pos_weight);
                    }
                    
                    // Create weighted loss function
                    torch::nn::BCEWithLogitsLoss weighted_bce_loss(weighted_options);
                    
                    // Apply weighted loss
                    torch::Tensor weighted_loss = weighted_bce_loss(input, target);
                    
                    // Compute gradients if possible
                    if (weighted_loss.defined() && !weighted_loss.isnan().any().item<bool>() && 
                        !weighted_loss.isinf().any().item<bool>() && input.requires_grad()) {
                        weighted_loss.backward();
                    }
                }
            } catch (const std::exception &) {
                // Ignore exceptions from the weighted version
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}