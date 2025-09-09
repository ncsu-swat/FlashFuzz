#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to work with
        if (Size < 16) {
            return 0;
        }

        // Generate first tensor
        auto input_tensor = generate_tensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0;
        }

        // Decide whether to compare with scalar or tensor
        bool use_scalar = (Data[offset % Size] % 2) == 0;
        offset++;

        torch::Tensor result;
        
        if (use_scalar) {
            // Test with scalar comparison
            double scalar_value = generate_float_value(Data, Size, offset);
            
            // Test torch::ne with scalar
            result = torch::ne(input_tensor, scalar_value);
            
            // Also test the tensor method version
            auto result2 = input_tensor.ne(scalar_value);
            
            // Verify results are boolean tensors
            if (result.dtype() != torch::kBool || result2.dtype() != torch::kBool) {
                std::cerr << "Result should be boolean tensor" << std::endl;
            }
            
            // Test with different scalar types
            if (offset + 4 < Size) {
                int int_scalar = static_cast<int>(Data[offset] | (Data[offset+1] << 8) | 
                                                 (Data[offset+2] << 16) | (Data[offset+3] << 24));
                offset += 4;
                auto result_int = torch::ne(input_tensor, int_scalar);
                if (result_int.dtype() != torch::kBool) {
                    std::cerr << "Integer scalar result should be boolean" << std::endl;
                }
            }
        } else {
            // Test with tensor comparison
            auto other_tensor = generate_tensor(Data, Size, offset);
            if (other_tensor.numel() == 0) {
                return 0;
            }
            
            try {
                // Test torch::ne with tensor
                result = torch::ne(input_tensor, other_tensor);
                
                // Also test the tensor method version
                auto result2 = input_tensor.ne(other_tensor);
                
                // Verify results are boolean tensors
                if (result.dtype() != torch::kBool || result2.dtype() != torch::kBool) {
                    std::cerr << "Result should be boolean tensor" << std::endl;
                }
                
                // Test broadcasting scenarios
                if (input_tensor.dim() > 0 && other_tensor.dim() > 0) {
                    // Try to create broadcastable tensors
                    auto shape1 = input_tensor.sizes().vec();
                    auto shape2 = other_tensor.sizes().vec();
                    
                    // Test with reshaped tensors for broadcasting
                    if (shape1.size() > 1) {
                        auto reshaped1 = input_tensor.view({-1, 1});
                        auto broadcast_result = torch::ne(reshaped1, other_tensor);
                        if (broadcast_result.dtype() != torch::kBool) {
                            std::cerr << "Broadcast result should be boolean" << std::endl;
                        }
                    }
                }
                
            } catch (const c10::Error& e) {
                // Broadcasting might fail, which is expected for incompatible shapes
                return 0;
            }
        }
        
        // Test with output tensor parameter
        if (result.defined() && result.numel() > 0) {
            auto out_tensor = torch::empty_like(result, torch::kBool);
            
            if (use_scalar) {
                double scalar_value = generate_float_value(Data, Size, offset);
                torch::ne_out(out_tensor, input_tensor, scalar_value);
            } else {
                auto other_tensor = generate_tensor(Data, Size, offset);
                if (other_tensor.numel() > 0) {
                    try {
                        torch::ne_out(out_tensor, input_tensor, other_tensor);
                    } catch (const c10::Error& e) {
                        // Shape mismatch is expected for some cases
                        return 0;
                    }
                }
            }
            
            // Verify output tensor has correct dtype
            if (out_tensor.dtype() != torch::kBool) {
                std::cerr << "Output tensor should be boolean" << std::endl;
            }
        }
        
        // Test edge cases with special values
        if (input_tensor.dtype().isFloatingType()) {
            // Test with NaN, inf, -inf
            auto nan_result = torch::ne(input_tensor, std::numeric_limits<double>::quiet_NaN());
            auto inf_result = torch::ne(input_tensor, std::numeric_limits<double>::infinity());
            auto neg_inf_result = torch::ne(input_tensor, -std::numeric_limits<double>::infinity());
            
            if (nan_result.dtype() != torch::kBool || 
                inf_result.dtype() != torch::kBool || 
                neg_inf_result.dtype() != torch::kBool) {
                std::cerr << "Special value results should be boolean" << std::endl;
            }
        }
        
        // Test with zero-dimensional tensors
        auto scalar_tensor = torch::tensor(42.0);
        auto zero_dim_result = torch::ne(scalar_tensor, input_tensor);
        if (zero_dim_result.dtype() != torch::kBool) {
            std::cerr << "Zero-dim result should be boolean" << std::endl;
        }
        
        // Test self-comparison (should be all False)
        auto self_result = torch::ne(input_tensor, input_tensor);
        if (self_result.dtype() != torch::kBool) {
            std::cerr << "Self-comparison result should be boolean" << std::endl;
        }
        
        // Verify that self-comparison gives all False (except for NaN values)
        if (input_tensor.dtype().isFloatingType()) {
            auto has_nan = torch::isnan(input_tensor).any().item<bool>();
            if (!has_nan) {
                auto all_false = torch::all(torch::logical_not(self_result)).item<bool>();
                if (!all_false) {
                    std::cerr << "Self-comparison without NaN should be all False" << std::endl;
                }
            }
        } else {
            // For non-floating types, self-comparison should always be all False
            auto all_false = torch::all(torch::logical_not(self_result)).item<bool>();
            if (!all_false) {
                std::cerr << "Self-comparison should be all False for non-floating types" << std::endl;
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}