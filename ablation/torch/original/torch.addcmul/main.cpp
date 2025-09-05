#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for creating tensors and value parameter
        if (Size < 10) {
            return 0;  // Not enough data, but don't discard
        }

        // Create three tensors for addcmul operation
        torch::Tensor input, tensor1, tensor2;
        
        // Parse first tensor (input)
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create the first tensor, try with a default one
            input = torch::randn({2, 3});
        }
        
        // Parse second tensor (tensor1)
        try {
            if (offset < Size) {
                tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create a tensor that can broadcast with input
                tensor1 = torch::randn({1});
            }
        } catch (const std::exception& e) {
            tensor1 = torch::randn({1});
        }
        
        // Parse third tensor (tensor2)
        try {
            if (offset < Size) {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create a tensor that can broadcast with input
                tensor2 = torch::randn({1});
            }
        } catch (const std::exception& e) {
            tensor2 = torch::randn({1});
        }
        
        // Parse value parameter
        double value = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Sanitize value to avoid extreme values
            if (!std::isfinite(value)) {
                value = 1.0;
            } else if (std::abs(value) > 1e6) {
                value = std::fmod(value, 1000.0);
            }
        } else if (offset < Size) {
            // Use remaining byte(s) to generate a simple value
            value = static_cast<double>(Data[offset] % 100) / 10.0 - 5.0;
            offset++;
        }
        
        // Determine if we should test with output tensor
        bool use_out_tensor = false;
        if (offset < Size) {
            use_out_tensor = (Data[offset++] % 2) == 0;
        }
        
        // Try to make tensors broadcastable by adjusting shapes if needed
        // This increases the chance of successful operation
        if (offset < Size && (Data[offset++] % 3) == 0) {
            // Sometimes make all tensors scalar for guaranteed broadcasting
            if (input.dim() > 0) input = input.reshape({-1})[0];
            if (tensor1.dim() > 0) tensor1 = tensor1.reshape({-1})[0];
            if (tensor2.dim() > 0) tensor2 = tensor2.reshape({-1})[0];
        }
        
        // Convert tensors to compatible dtypes if needed
        // addcmul requires compatible types for arithmetic
        torch::ScalarType target_dtype = input.scalar_type();
        
        // For integer types with floating point value, convert to float
        if ((target_dtype == torch::kInt8 || target_dtype == torch::kUInt8 || 
             target_dtype == torch::kInt16 || target_dtype == torch::kInt32 || 
             target_dtype == torch::kInt64) && value != std::floor(value)) {
            input = input.to(torch::kFloat32);
            target_dtype = torch::kFloat32;
        }
        
        // Ensure all tensors have compatible types
        if (tensor1.scalar_type() != target_dtype) {
            tensor1 = tensor1.to(target_dtype);
        }
        if (tensor2.scalar_type() != target_dtype) {
            tensor2 = tensor2.to(target_dtype);
        }
        
        torch::Tensor result;
        
        if (use_out_tensor) {
            // Test with pre-allocated output tensor
            try {
                // Create output tensor with broadcast shape
                torch::Tensor out;
                
                // Try to infer the broadcast shape
                try {
                    auto broadcast_shape = torch::broadcast_tensors({input, tensor1, tensor2})[0].sizes();
                    out = torch::empty(broadcast_shape, input.options());
                } catch (...) {
                    // If broadcast fails, use input shape
                    out = torch::empty_like(input);
                }
                
                result = torch::addcmul(input, tensor1, tensor2, value, out);
                
                // Verify that result and out are the same tensor
                if (result.data_ptr() != out.data_ptr()) {
                    std::cerr << "Warning: out parameter not used correctly" << std::endl;
                }
            } catch (const c10::Error& e) {
                // Try without out parameter if it fails
                result = torch::addcmul(input, tensor1, tensor2, value);
            }
        } else {
            // Test without output tensor
            result = torch::addcmul(input, tensor1, tensor2, value);
        }
        
        // Perform various checks on the result
        if (result.defined()) {
            // Check for NaN or Inf in result
            if (result.is_floating_point() || result.is_complex()) {
                auto has_nan = torch::isnan(result).any().item<bool>();
                auto has_inf = torch::isinf(result).any().item<bool>();
                
                if (has_nan && !torch::isnan(input).any().item<bool>() && 
                    !torch::isnan(tensor1).any().item<bool>() && 
                    !torch::isnan(tensor2).any().item<bool>()) {
                    // NaN introduced by operation
                    #ifdef DEBUG_FUZZ
                    std::cout << "NaN introduced by addcmul operation" << std::endl;
                    #endif
                }
            }
            
            // Verify the operation manually on a few elements if tensors are small
            if (result.numel() > 0 && result.numel() <= 10) {
                // Manual verification for small tensors
                auto input_flat = input.flatten();
                auto tensor1_flat = tensor1.flatten();
                auto tensor2_flat = tensor2.flatten();
                auto result_flat = result.flatten();
                
                // Only verify if all tensors have compatible sizes after flattening
                if (input_flat.numel() == result_flat.numel()) {
                    // The actual formula: out = input + value * tensor1 * tensor2
                    // Note: broadcasting might make this complex, so we just check it doesn't crash
                    auto manual_result = input + value * tensor1 * tensor2;
                    
                    if (!torch::allclose(result, manual_result, 1e-5, 1e-8)) {
                        #ifdef DEBUG_FUZZ
                        std::cout << "Result mismatch with manual calculation" << std::endl;
                        #endif
                    }
                }
            }
            
            // Test in-place variant if shapes are compatible
            if (offset < Size && (Data[offset] % 4) == 0) {
                try {
                    auto input_copy = input.clone();
                    input_copy.addcmul_(tensor1, tensor2, value);
                    
                    if (!torch::allclose(input_copy, result, 1e-5, 1e-8)) {
                        #ifdef DEBUG_FUZZ
                        std::cout << "In-place variant produces different result" << std::endl;
                        #endif
                    }
                } catch (const c10::Error& e) {
                    // In-place might fail due to shape/type constraints
                    #ifdef DEBUG_FUZZ
                    std::cout << "In-place variant failed: " << e.what() << std::endl;
                    #endif
                }
            }
        }
        
        // Test edge cases based on remaining fuzzer input
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 5) {
                case 0:
                    // Test with zero value
                    result = torch::addcmul(input, tensor1, tensor2, 0.0);
                    // Result should equal input
                    if (!torch::allclose(result, input, 1e-5, 1e-8)) {
                        #ifdef DEBUG_FUZZ
                        std::cout << "Zero value doesn't preserve input" << std::endl;
                        #endif
                    }
                    break;
                case 1:
                    // Test with negative value
                    result = torch::addcmul(input, tensor1, tensor2, -value);
                    break;
                case 2:
                    // Test with very large value
                    result = torch::addcmul(input, tensor1, tensor2, 1e10);
                    break;
                case 3:
                    // Test with very small value
                    result = torch::addcmul(input, tensor1, tensor2, 1e-10);
                    break;
                case 4:
                    // Test with integer value for integer tensors
                    if (input.scalar_type() == torch::kInt32 || 
                        input.scalar_type() == torch::kInt64) {
                        result = torch::addcmul(input, tensor1, tensor2, 
                                               static_cast<int64_t>(value));
                    }
                    break;
            }
        }
        
        return 0;
    }
    catch (const c10::Error &e)
    {
        // PyTorch errors are expected for invalid operations
        #ifdef DEBUG_FUZZ
        std::cout << "PyTorch error: " << e.what() << std::endl;
        #endif
        return 0;  // Continue fuzzing
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}