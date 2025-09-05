#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal data for tensor creation and parameters
        if (Size < 4) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse RReLU parameters from remaining bytes
        double lower = 0.125;  // Default lower bound
        double upper = 0.333;  // Default upper bound
        bool training = false;
        bool inplace = false;
        
        // Parse lower bound if we have data
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t lower_byte = Data[offset++];
            // Map to range [0.0, 1.0] with some interesting values
            lower = static_cast<double>(lower_byte) / 255.0;
        }
        
        // Parse upper bound if we have data
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t upper_byte = Data[offset++];
            // Map to range [0.0, 1.0]
            upper = static_cast<double>(upper_byte) / 255.0;
        }
        
        // Ensure lower <= upper for valid range
        if (lower > upper) {
            std::swap(lower, upper);
        }
        
        // Parse training flag
        if (offset + sizeof(uint8_t) <= Size) {
            training = (Data[offset++] & 0x01) != 0;
        }
        
        // Parse inplace flag
        if (offset + sizeof(uint8_t) <= Size) {
            inplace = (Data[offset++] & 0x01) != 0;
        }
        
        // Test various edge cases and configurations
        try {
            // Test 1: Basic rrelu with parsed parameters
            torch::Tensor result1;
            if (inplace && input.is_floating_point()) {
                // Clone for inplace operation to avoid modifying original
                torch::Tensor input_copy = input.clone();
                result1 = torch::nn::functional::rrelu(input_copy, 
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(true));
            } else {
                result1 = torch::nn::functional::rrelu(input,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
            
            // Test 2: Try with different training modes
            torch::Tensor result2 = torch::nn::functional::rrelu(input,
                torch::nn::functional::RReLUFuncOptions()
                    .lower(lower)
                    .upper(upper)
                    .training(!training)
                    .inplace(false));
            
            // Test 3: Edge case with extreme values
            torch::Tensor result3 = torch::nn::functional::rrelu(input,
                torch::nn::functional::RReLUFuncOptions()
                    .lower(0.0)
                    .upper(1.0)
                    .training(true)
                    .inplace(false));
            
            // Test 4: Equal lower and upper (degenerates to leaky ReLU)
            torch::Tensor result4 = torch::nn::functional::rrelu(input,
                torch::nn::functional::RReLUFuncOptions()
                    .lower(0.2)
                    .upper(0.2)
                    .training(false)
                    .inplace(false));
            
            // Test 5: Try with zero tensor
            if (input.numel() > 0) {
                torch::Tensor zero_tensor = torch::zeros_like(input);
                torch::Tensor result5 = torch::nn::functional::rrelu(zero_tensor,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
            
            // Test 6: Try with ones tensor
            if (input.numel() > 0) {
                torch::Tensor ones_tensor = torch::ones_like(input);
                torch::Tensor result6 = torch::nn::functional::rrelu(ones_tensor,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
            
            // Test 7: Try with negative values
            if (input.numel() > 0 && input.is_floating_point()) {
                torch::Tensor neg_tensor = -torch::abs(input);
                torch::Tensor result7 = torch::nn::functional::rrelu(neg_tensor,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
            
            // Test 8: Mixed positive/negative values
            if (input.numel() > 1 && input.is_floating_point()) {
                torch::Tensor mixed_tensor = input - input.mean();
                torch::Tensor result8 = torch::nn::functional::rrelu(mixed_tensor,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
            
            // Test 9: Test with reshaped tensor
            if (input.numel() > 0) {
                torch::Tensor flat = input.flatten();
                torch::Tensor result9 = torch::nn::functional::rrelu(flat,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
            
            // Test 10: Test with permuted tensor (non-contiguous)
            if (input.dim() >= 2) {
                torch::Tensor permuted = input.transpose(0, -1);
                torch::Tensor result10 = torch::nn::functional::rrelu(permuted,
                    torch::nn::functional::RReLUFuncOptions()
                        .lower(lower)
                        .upper(upper)
                        .training(training)
                        .inplace(false));
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors - these are expected for invalid operations
            // Continue fuzzing
            return 0;
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid tensor operations
            // Continue fuzzing
            return 0;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}