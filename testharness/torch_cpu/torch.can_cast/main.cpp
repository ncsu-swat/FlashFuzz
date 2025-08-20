#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for the data types
        if (Size < 2) {
            return 0;
        }
        
        // Parse the source and destination data types
        uint8_t src_dtype_selector = Data[offset++];
        uint8_t dst_dtype_selector = Data[offset++];
        
        // Get the actual ScalarType values
        torch::ScalarType src_dtype = fuzzer_utils::parseDataType(src_dtype_selector);
        torch::ScalarType dst_dtype = fuzzer_utils::parseDataType(dst_dtype_selector);
        
        // Test can_cast function
        bool can_cast_result = torch::can_cast(src_dtype, dst_dtype);
        
        // Create a tensor with the source data type to test with actual data
        if (offset < Size) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Only attempt to cast if the tensor is valid and can_cast returned true
                if (can_cast_result) {
                    // Try to cast the tensor to the destination type
                    torch::Tensor cast_tensor = tensor.to(dst_dtype);
                }
            } catch (const std::exception& e) {
                // Catch exceptions from tensor creation or casting
                // This is expected in some cases, so we continue
            }
        }
        
        // Test with empty tensor
        if (can_cast_result) {
            torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(src_dtype));
            torch::Tensor cast_empty = empty_tensor.to(dst_dtype);
        }
        
        // Test with scalar tensor
        if (can_cast_result) {
            torch::Tensor scalar_tensor = torch::scalar_tensor(1, torch::TensorOptions().dtype(src_dtype));
            torch::Tensor cast_scalar = scalar_tensor.to(dst_dtype);
        }
        
        // Test with tensor containing extreme values
        if (can_cast_result) {
            // Create tensors with extreme values based on source type
            torch::Tensor extreme_tensor;
            
            if (src_dtype == torch::kFloat || src_dtype == torch::kDouble) {
                extreme_tensor = torch::tensor({std::numeric_limits<float>::max(), 
                                               std::numeric_limits<float>::min(),
                                               std::numeric_limits<float>::infinity(),
                                               -std::numeric_limits<float>::infinity(),
                                               std::numeric_limits<float>::quiet_NaN()}, 
                                               torch::TensorOptions().dtype(src_dtype));
            } else if (src_dtype == torch::kInt32 || src_dtype == torch::kInt64) {
                extreme_tensor = torch::tensor({std::numeric_limits<int32_t>::max(), 
                                               std::numeric_limits<int32_t>::min()}, 
                                               torch::TensorOptions().dtype(src_dtype));
            } else if (src_dtype == torch::kBool) {
                extreme_tensor = torch::tensor({true, false}, 
                                               torch::TensorOptions().dtype(src_dtype));
            } else {
                // For other types, create a simple tensor
                extreme_tensor = torch::ones({2}, torch::TensorOptions().dtype(src_dtype));
            }
            
            // Try to cast the extreme tensor
            torch::Tensor cast_extreme = extreme_tensor.to(dst_dtype);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}