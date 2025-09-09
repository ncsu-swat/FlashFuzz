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

        // Extract tensor configuration parameters
        auto shape_info = extract_tensor_shape(Data, Size, offset);
        if (shape_info.empty()) {
            return 0;
        }

        auto dtype = extract_dtype(Data, Size, offset);
        auto device = extract_device(Data, Size, offset);

        // Create base tensor
        torch::Tensor base_tensor;
        try {
            base_tensor = create_tensor(shape_info, dtype, device, Data, Size, offset);
        } catch (...) {
            return 0;
        }

        // Test different pow variants
        
        // 1. Test tensor.pow(scalar)
        if (offset + 8 <= Size) {
            double scalar_exp = extract_float_value(Data, Size, offset);
            
            // Handle special cases and edge values
            if (std::isnan(scalar_exp)) {
                scalar_exp = 2.0;
            }
            if (std::isinf(scalar_exp)) {
                scalar_exp = std::signbit(scalar_exp) ? -2.0 : 2.0;
            }
            
            // Clamp extreme values to prevent overflow
            scalar_exp = std::max(-100.0, std::min(100.0, scalar_exp));
            
            auto result1 = torch::pow(base_tensor, scalar_exp);
            
            // Test in-place version if possible
            if (base_tensor.is_floating_point() && !base_tensor.requires_grad()) {
                auto base_copy = base_tensor.clone();
                base_copy.pow_(scalar_exp);
            }
        }

        // 2. Test tensor.pow(tensor)
        if (offset < Size) {
            // Create exponent tensor with same or broadcastable shape
            auto exp_shape = shape_info;
            
            // Sometimes use scalar tensor (single element)
            if (extract_bool(Data, Size, offset)) {
                exp_shape = {1};
            }
            
            // Sometimes use different but broadcastable shape
            if (extract_bool(Data, Size, offset) && !exp_shape.empty()) {
                // Make some dimensions 1 for broadcasting
                for (size_t i = 0; i < exp_shape.size() && i < 2; ++i) {
                    if (extract_bool(Data, Size, offset)) {
                        exp_shape[i] = 1;
                    }
                }
            }

            torch::Tensor exp_tensor;
            try {
                exp_tensor = create_tensor(exp_shape, dtype, device, Data, Size, offset);
                
                // Clamp exponent values to prevent extreme results
                if (exp_tensor.is_floating_point()) {
                    exp_tensor = torch::clamp(exp_tensor, -10.0, 10.0);
                } else if (exp_tensor.is_signed()) {
                    exp_tensor = torch::clamp(exp_tensor, -10, 10);
                } else {
                    exp_tensor = torch::clamp(exp_tensor, 0, 10);
                }
                
                auto result2 = torch::pow(base_tensor, exp_tensor);
                
                // Test in-place version
                if (base_tensor.is_floating_point() && !base_tensor.requires_grad() && 
                    base_tensor.sizes() == exp_tensor.sizes()) {
                    auto base_copy = base_tensor.clone();
                    base_copy.pow_(exp_tensor);
                }
            } catch (...) {
                // Skip if tensor creation fails
            }
        }

        // 3. Test torch.pow(scalar, tensor) - scalar base, tensor exponent
        if (offset + 8 <= Size) {
            double scalar_base = extract_float_value(Data, Size, offset);
            
            // Handle special cases
            if (std::isnan(scalar_base)) {
                scalar_base = 2.0;
            }
            if (std::isinf(scalar_base)) {
                scalar_base = std::signbit(scalar_base) ? -2.0 : 2.0;
            }
            
            // Avoid extreme bases that could cause overflow
            scalar_base = std::max(-10.0, std::min(10.0, scalar_base));
            
            // Use base_tensor as exponent (already clamped above)
            auto clamped_exp = base_tensor.clone();
            if (clamped_exp.is_floating_point()) {
                clamped_exp = torch::clamp(clamped_exp, -5.0, 5.0);
            } else if (clamped_exp.is_signed()) {
                clamped_exp = torch::clamp(clamped_exp, -5, 5);
            } else {
                clamped_exp = torch::clamp(clamped_exp, 0, 5);
            }
            
            auto result3 = torch::pow(scalar_base, clamped_exp);
        }

        // 4. Test with different tensor types and edge cases
        if (offset < Size) {
            // Test with integer tensors
            if (!base_tensor.is_floating_point()) {
                // For integer tensors, use small positive exponents
                auto small_exp = torch::randint(0, 4, base_tensor.sizes(), 
                                              torch::TensorOptions().dtype(torch::kInt32).device(device));
                auto result4 = torch::pow(base_tensor, small_exp);
            }
            
            // Test with complex numbers if supported
            if (extract_bool(Data, Size, offset) && base_tensor.is_floating_point()) {
                try {
                    auto complex_base = torch::complex(base_tensor, torch::zeros_like(base_tensor));
                    auto result5 = torch::pow(complex_base, 2.0);
                } catch (...) {
                    // Complex operations might not be supported in all configurations
                }
            }
        }

        // 5. Test edge cases with special values
        if (base_tensor.is_floating_point() && offset < Size) {
            auto special_base = base_tensor.clone();
            
            // Test with some special values
            if (extract_bool(Data, Size, offset)) {
                special_base.fill_(0.0);  // 0^x cases
            } else if (extract_bool(Data, Size, offset)) {
                special_base.fill_(1.0);  // 1^x cases
            } else if (extract_bool(Data, Size, offset)) {
                special_base.fill_(-1.0); // (-1)^x cases
            }
            
            double special_exp = extract_bool(Data, Size, offset) ? 0.0 : 
                               (extract_bool(Data, Size, offset) ? 1.0 : 2.0);
            
            auto result6 = torch::pow(special_base, special_exp);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}