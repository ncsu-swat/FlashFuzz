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
        
        // Create input, variance, and target tensors
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the next tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor variance = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the next tensor
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor target = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input, variance, and target have the same shape
        // If not, we'll reshape them to match the smallest one
        std::vector<int64_t> common_shape;
        if (input.dim() > 0 && variance.dim() > 0 && target.dim() > 0) {
            // Find the tensor with the smallest number of dimensions
            int min_dim = std::min({input.dim(), variance.dim(), target.dim()});
            
            // Get the shape of the tensor with the smallest dimensions
            if (input.dim() == min_dim) {
                common_shape = input.sizes().vec();
            } else if (variance.dim() == min_dim) {
                common_shape = variance.sizes().vec();
            } else {
                common_shape = target.sizes().vec();
            }
            
            // Reshape tensors if needed
            if (input.dim() != min_dim || input.sizes() != common_shape) {
                if (input.numel() > 0) {
                    input = input.reshape(common_shape);
                } else {
                    input = torch::zeros(common_shape);
                }
            }
            
            if (variance.dim() != min_dim || variance.sizes() != common_shape) {
                if (variance.numel() > 0) {
                    variance = variance.reshape(common_shape);
                } else {
                    variance = torch::ones(common_shape);
                }
            }
            
            if (target.dim() != min_dim || target.sizes() != common_shape) {
                if (target.numel() > 0) {
                    target = target.reshape(common_shape);
                } else {
                    target = torch::zeros(common_shape);
                }
            }
        } else {
            // Handle the case where at least one tensor has 0 dimensions
            if (input.dim() == 0) {
                input = torch::tensor(1.0f);
            }
            if (variance.dim() == 0) {
                variance = torch::tensor(1.0f);
            }
            if (target.dim() == 0) {
                target = torch::tensor(0.0f);
            }
            
            // Ensure all tensors have the same shape
            common_shape = {1};
            input = input.reshape(common_shape);
            variance = variance.reshape(common_shape);
            target = target.reshape(common_shape);
        }
        
        // Ensure variance is positive (required by GaussianNLLLoss)
        variance = torch::abs(variance) + 1e-6;
        
        // Parse options for GaussianNLLLoss
        bool full = false;
        double eps = 1e-6;
        torch::Reduction::Reduction reduction = torch::Reduction::Mean;
        
        // Use remaining data to set options if available
        if (offset < Size) {
            full = Data[offset++] % 2 == 0;
        }
        
        if (offset < Size) {
            uint8_t red_val = Data[offset++] % 3;
            switch (red_val) {
                case 0: reduction = torch::Reduction::None; break;
                case 1: reduction = torch::Reduction::Sum; break;
                case 2: 
                default: reduction = torch::Reduction::Mean; break;
            }
        }
        
        // Apply the Gaussian NLL loss function directly
        torch::Tensor loss = torch::nn::functional::gaussian_nll_loss(
            input, target, variance,
            torch::nn::functional::GaussianNLLLossFuncOptions()
                .full(full)
                .eps(eps)
                .reduction(reduction)
        );
        
        // Ensure the loss is finite
        if (loss.defined() && !loss.isfinite().all().item<bool>()) {
            // This is not an error, just a case we want to note
            return 0;
        }
        
        // Test backward pass if possible
        if (input.requires_grad() && loss.numel() > 0 && loss.scalar_type() != torch::kHalf) {
            try {
                loss.backward();
            } catch (const std::exception& e) {
                // Backward pass failed, but this is not an error for the fuzzer
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