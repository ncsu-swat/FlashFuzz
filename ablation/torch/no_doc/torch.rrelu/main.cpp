#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

// Helper to consume a value from fuzzer data
template<typename T>
T consume(const uint8_t* &data, size_t &remaining) {
    if (remaining < sizeof(T)) {
        remaining = 0;
        return T{};
    }
    T value;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    remaining -= sizeof(T);
    return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    if (Size < 4) {
        return 0; // Need minimum bytes for basic parameters
    }

    try
    {
        size_t offset = 0;
        
        // Create input tensor with diverse properties
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Consume remaining bytes for rrelu parameters
        const uint8_t* remaining_data = Data + offset;
        size_t remaining_size = (offset < Size) ? Size - offset : 0;
        
        // Parse lower bound (typically 1/8)
        double lower = 0.125; // default
        if (remaining_size >= sizeof(float)) {
            float raw_lower = consume<float>(remaining_data, remaining_size);
            // Map to reasonable range [0, 0.5] to avoid invalid configurations
            lower = std::abs(raw_lower);
            lower = std::fmod(lower, 0.5);
            if (lower < 0.001) lower = 0.001; // Avoid exactly 0
        }
        
        // Parse upper bound (typically 1/3)
        double upper = 0.3333; // default
        if (remaining_size >= sizeof(float)) {
            float raw_upper = consume<float>(remaining_data, remaining_size);
            // Ensure upper > lower, map to range [lower, 1.0]
            upper = std::abs(raw_upper);
            upper = lower + std::fmod(upper, 1.0 - lower);
            if (upper <= lower) upper = lower + 0.01;
            if (upper > 1.0) upper = 1.0;
        }
        
        // Parse training mode flag
        bool training = true;
        if (remaining_size >= 1) {
            training = consume<uint8_t>(remaining_data, remaining_size) & 1;
        }
        
        // Parse inplace flag
        bool inplace = false;
        if (remaining_size >= 1) {
            inplace = consume<uint8_t>(remaining_data, remaining_size) & 1;
        }
        
        // Test various tensor configurations
        if (remaining_size >= 1) {
            uint8_t config = consume<uint8_t>(remaining_data, remaining_size);
            
            // Try different memory layouts
            if (config & 0x01) {
                // Make non-contiguous via transpose if applicable
                if (input.dim() >= 2) {
                    input = input.transpose(0, 1);
                }
            }
            
            // Try different devices if available
            if ((config & 0x02) && torch::cuda::is_available()) {
                input = input.to(torch::kCUDA);
            }
            
            // Try requires_grad variations
            if (config & 0x04) {
                // Only set requires_grad for floating point types
                if (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble ||
                    input.dtype() == torch::kHalf || input.dtype() == torch::kBFloat16) {
                    input.requires_grad_(true);
                }
            }
            
            // Try sparse tensors for 2D inputs
            if ((config & 0x08) && input.dim() == 2 && input.numel() > 0) {
                try {
                    // Create sparse tensor from dense
                    auto sparse_input = input.to_sparse();
                    // Apply rrelu to sparse tensor
                    torch::nn::functional::rrelu(sparse_input, 
                        torch::nn::functional::RReLUFuncOptions()
                            .lower(lower)
                            .upper(upper)
                            .training(training)
                            .inplace(false)); // Can't do inplace on sparse
                } catch (...) {
                    // Sparse operation might not be supported, continue
                }
            }
        }
        
        // Main rrelu operation
        torch::Tensor output;
        
        if (inplace && input.numel() > 0) {
            // Try in-place operation
            try {
                // Clone for comparison if needed
                torch::Tensor input_copy = input.clone();
                
                // In-place rrelu modifies input directly
                torch::nn::functional::rrelu(input,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(true));
                output = input;
                
                // Verify in-place modification occurred
                if (input.data_ptr() != output.data_ptr()) {
                    std::cerr << "Warning: In-place operation created new tensor" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Some dtypes might not support in-place, try regular
                output = torch::nn::functional::rrelu(input,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
        } else {
            // Regular out-of-place operation
            output = torch::nn::functional::rrelu(input,
                torch::nn::functional::RReLUFuncOptions()
                    .lower(lower)
                    .upper(upper)
                    .training(training)
                    .inplace(false));
        }
        
        // Additional edge case testing
        if (remaining_size >= 1) {
            uint8_t extra_tests = consume<uint8_t>(remaining_data, remaining_size);
            
            // Test with extreme values
            if (extra_tests & 0x01) {
                torch::Tensor extreme_input = torch::full_like(input, std::numeric_limits<float>::max());
                torch::nn::functional::rrelu(extreme_input,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(false));
            }
            
            // Test with NaN/Inf values for floating types
            if ((extra_tests & 0x02) && 
                (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble)) {
                torch::Tensor special_input = input.clone();
                if (special_input.numel() > 0) {
                    special_input[0] = std::numeric_limits<float>::quiet_NaN();
                    if (special_input.numel() > 1) {
                        special_input[1] = std::numeric_limits<float>::infinity();
                    }
                    torch::nn::functional::rrelu(special_input,
                        torch::nn::functional::RReLUFuncOptions()
                            .lower(lower)
                            .upper(upper)
                            .training(training));
                }
            }
            
            // Test gradient computation if applicable
            if ((extra_tests & 0x04) && input.requires_grad()) {
                try {
                    auto grad_output = output.sum();
                    grad_output.backward();
                    // Access gradient to ensure it was computed
                    auto grad = input.grad();
                } catch (...) {
                    // Gradient computation might fail for some configurations
                }
            }
            
            // Test with zero-sized dimensions
            if (extra_tests & 0x08) {
                try {
                    torch::Tensor zero_tensor = torch::empty({0, 5}, input.options());
                    torch::nn::functional::rrelu(zero_tensor,
                        torch::nn::functional::RReLUFuncOptions()
                            .lower(lower)
                            .upper(upper)
                            .training(training));
                } catch (...) {
                    // Some configurations might not handle zero dimensions
                }
            }
        }
        
        // Validate output properties
        if (output.defined()) {
            // Check shape preservation
            if (output.sizes() != input.sizes()) {
                std::cerr << "Shape changed: " << input.sizes() << " -> " << output.sizes() << std::endl;
            }
            
            // In eval mode, verify deterministic behavior
            if (!training && input.numel() > 0) {
                torch::Tensor output2 = torch::nn::functional::rrelu(input,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(false)
                        .inplace(false));
                
                // Should be identical in eval mode
                if (!torch::allclose(output, output2, 1e-5, 1e-8)) {
                    std::cerr << "Non-deterministic behavior in eval mode" << std::endl;
                }
            }
        }
        
    }
    catch (const c10::Error& e)
    {
        // PyTorch-specific errors are expected during fuzzing
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}