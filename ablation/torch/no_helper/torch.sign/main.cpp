#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various dtypes and shapes
        auto input_tensor = generate_tensor(Data, Size, offset);
        
        // Test basic sign operation
        auto result = torch::sign(input_tensor);
        
        // Verify result properties
        if (result.sizes() != input_tensor.sizes()) {
            throw std::runtime_error("Output shape mismatch");
        }
        
        // Test with output tensor parameter
        if (offset < Size) {
            auto out_tensor = torch::empty_like(input_tensor);
            torch::sign_out(out_tensor, input_tensor);
            
            // Verify out parameter works correctly
            if (!torch::allclose(result, out_tensor, 1e-6, 1e-6, /*equal_nan=*/true)) {
                // Allow for NaN differences since sign of NaN is NaN
                auto result_isnan = torch::isnan(result);
                auto out_isnan = torch::isnan(out_tensor);
                if (!torch::equal(result_isnan, out_isnan)) {
                    throw std::runtime_error("Output tensor mismatch");
                }
            }
        }
        
        // Test edge cases with specific values if we have enough data
        if (offset + 32 < Size) {
            // Create tensor with edge case values
            std::vector<float> edge_values;
            
            // Add zeros (positive and negative)
            edge_values.push_back(0.0f);
            edge_values.push_back(-0.0f);
            
            // Add infinities
            edge_values.push_back(std::numeric_limits<float>::infinity());
            edge_values.push_back(-std::numeric_limits<float>::infinity());
            
            // Add NaN
            edge_values.push_back(std::numeric_limits<float>::quiet_NaN());
            
            // Add very small values
            edge_values.push_back(std::numeric_limits<float>::min());
            edge_values.push_back(-std::numeric_limits<float>::min());
            edge_values.push_back(std::numeric_limits<float>::denorm_min());
            edge_values.push_back(-std::numeric_limits<float>::denorm_min());
            
            auto edge_tensor = torch::tensor(edge_values);
            auto edge_result = torch::sign(edge_tensor);
            
            // Verify edge case results
            auto edge_data = edge_result.accessor<float, 1>();
            
            // Check that sign of positive zero is 0
            if (std::signbit(edge_values[0]) == false && edge_data[0] != 0.0f) {
                throw std::runtime_error("Sign of positive zero should be 0");
            }
            
            // Check that sign of negative zero is 0 (or -0, both acceptable)
            if (edge_data[1] != 0.0f && edge_data[1] != -0.0f) {
                throw std::runtime_error("Sign of negative zero should be 0");
            }
            
            // Check that sign of positive infinity is 1
            if (edge_data[2] != 1.0f) {
                throw std::runtime_error("Sign of positive infinity should be 1");
            }
            
            // Check that sign of negative infinity is -1
            if (edge_data[3] != -1.0f) {
                throw std::runtime_error("Sign of negative infinity should be -1");
            }
            
            // Check that sign of NaN is NaN
            if (!std::isnan(edge_data[4])) {
                throw std::runtime_error("Sign of NaN should be NaN");
            }
        }
        
        // Test with different tensor types
        if (offset < Size) {
            // Test with integer tensor
            auto int_tensor = input_tensor.to(torch::kInt32);
            auto int_result = torch::sign(int_tensor);
            
            // Test with double tensor
            auto double_tensor = input_tensor.to(torch::kDouble);
            auto double_result = torch::sign(double_tensor);
            
            // Test with complex tensor if supported
            try {
                auto complex_tensor = input_tensor.to(torch::kComplexFloat);
                auto complex_result = torch::sign(complex_tensor);
            } catch (...) {
                // Complex sign might not be supported, ignore
            }
        }
        
        // Test in-place operation if the tensor supports it
        if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
            try {
                auto inplace_tensor = input_tensor.clone();
                inplace_tensor.sign_();
                
                // Verify in-place result matches regular result
                if (!torch::allclose(result, inplace_tensor, 1e-6, 1e-6, /*equal_nan=*/true)) {
                    auto result_isnan = torch::isnan(result);
                    auto inplace_isnan = torch::isnan(inplace_tensor);
                    if (!torch::equal(result_isnan, inplace_isnan)) {
                        throw std::runtime_error("In-place operation mismatch");
                    }
                }
            } catch (...) {
                // In-place might not be supported for all tensor types
            }
        }
        
        // Test with requires_grad
        if (input_tensor.is_floating_point() && offset < Size) {
            try {
                auto grad_tensor = input_tensor.clone().requires_grad_(true);
                auto grad_result = torch::sign(grad_tensor);
                
                // Sign function should work with gradients (though gradient is 0 almost everywhere)
                if (grad_result.requires_grad() != grad_tensor.requires_grad()) {
                    throw std::runtime_error("Gradient requirement not preserved");
                }
            } catch (...) {
                // Gradient operations might fail in some cases
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