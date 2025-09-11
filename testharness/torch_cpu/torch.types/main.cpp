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
        
        // Skip if not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor to test torch.types with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch.types functionality
        auto dtype = tensor.dtype();
        auto scalar_type = tensor.scalar_type();
        
        // Get the type name as string
        std::string type_name = torch::toString(dtype);
        
        // Test type conversion
        if (offset + 1 < Size) {
            uint8_t target_type_selector = Data[offset++];
            auto target_type = fuzzer_utils::parseDataType(target_type_selector);
            
            // Convert tensor to the target type
            torch::Tensor converted = tensor.to(target_type);
            
            // Verify the conversion worked
            if (converted.dtype() != target_type) {
                throw std::runtime_error("Type conversion failed");
            }
            
            // Test type promotion with operations
            if (offset + 1 < Size) {
                // Create another tensor with potentially different type
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test type promotion in binary operations
                torch::Tensor result = tensor + tensor2;
                
                // Get the promoted type
                auto promoted_type = result.dtype();
            }
        }
        
        // Test is_floating_point, is_complex, etc.
        bool is_floating = tensor.is_floating_point();
        bool is_complex = tensor.is_complex();
        bool is_signed = tensor.is_signed();
        
        // Test type properties
        int64_t itemsize = tensor.element_size();
        
        // Test type casting
        if (tensor.numel() > 0) {
            // Try to access data with different type interpretations
            if (tensor.is_floating_point() && tensor.dtype() == torch::kFloat) {
                float value = tensor.item<float>();
            } else if (tensor.dtype() == torch::kInt64) {
                int64_t value = tensor.item<int64_t>();
            } else if (tensor.dtype() == torch::kBool) {
                bool value = tensor.item<bool>();
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
