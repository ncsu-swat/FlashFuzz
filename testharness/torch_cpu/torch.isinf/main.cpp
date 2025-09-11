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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.isinf operation
        torch::Tensor result = torch::isinf(input_tensor);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Create a boolean tensor to test the out parameter
            auto options = torch::TensorOptions().dtype(torch::kBool);
            torch::Tensor out_tensor = torch::empty_like(input_tensor, options);
            torch::isinf_out(out_tensor, input_tensor);
            
            // Test with nan=True parameter if available in the API
            if (offset + 2 < Size) {
                bool include_nan = Data[offset++] % 2 == 0;
                if (include_nan) {
                    // Note: torch::isinf doesn't have a nan parameter in C++ API
                    // This is just to show how we would handle additional parameters
                    torch::Tensor nan_result = torch::isinf(input_tensor);
                }
            }
        }
        
        // Test edge cases by creating special tensors with inf values
        if (offset + 3 < Size) {
            uint8_t special_case = Data[offset++] % 4;
            
            if (special_case == 0) {
                // Create tensor with inf values
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor inf_tensor = torch::full({2, 2}, std::numeric_limits<float>::infinity(), options);
                torch::Tensor inf_result = torch::isinf(inf_tensor);
            }
            else if (special_case == 1) {
                // Create tensor with -inf values
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor neg_inf_tensor = torch::full({2, 2}, -std::numeric_limits<float>::infinity(), options);
                torch::Tensor neg_inf_result = torch::isinf(neg_inf_tensor);
            }
            else if (special_case == 2) {
                // Create tensor with NaN values
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor nan_tensor = torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN(), options);
                torch::Tensor nan_result = torch::isinf(nan_tensor);
            }
            else {
                // Create tensor with mixed values (normal, inf, -inf, NaN)
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor mixed_tensor = torch::empty({2, 2}, options);
                mixed_tensor.index_put_({0, 0}, 1.0);
                mixed_tensor.index_put_({0, 1}, std::numeric_limits<float>::infinity());
                mixed_tensor.index_put_({1, 0}, -std::numeric_limits<float>::infinity());
                mixed_tensor.index_put_({1, 1}, std::numeric_limits<float>::quiet_NaN());
                torch::Tensor mixed_result = torch::isinf(mixed_tensor);
            }
        }
        
        // Test with empty tensor
        if (offset + 4 < Size) {
            auto options = torch::TensorOptions().dtype(torch::kFloat);
            torch::Tensor empty_tensor = torch::empty({0}, options);
            torch::Tensor empty_result = torch::isinf(empty_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
