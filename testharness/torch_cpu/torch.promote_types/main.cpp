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
        
        // Need at least 2 bytes for the scalar types
        if (Size < 2) {
            return 0;
        }
        
        // Parse two scalar types from the input data
        uint8_t type1_selector = Data[offset++];
        uint8_t type2_selector = Data[offset++];
        
        // Get the actual scalar types
        torch::ScalarType type1 = fuzzer_utils::parseDataType(type1_selector);
        torch::ScalarType type2 = fuzzer_utils::parseDataType(type2_selector);
        
        // Call promote_types to get the promoted type
        torch::ScalarType promoted_type = torch::promote_types(type1, type2);
        
        // Create tensors of the respective types to verify the promotion works in practice
        if (offset + 2 < Size) {
            // Create tensors with the original types
            torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Convert tensors to the specified types
            tensor1 = tensor1.to(type1);
            tensor2 = tensor2.to(type2);
            
            // Test the promotion by adding the tensors
            // This should automatically promote to the common type
            torch::Tensor result = tensor1 + tensor2;
            
            // Verify the result has the expected promoted type
            if (result.scalar_type() != promoted_type) {
                throw std::runtime_error("Promotion type mismatch: expected " + 
                                        std::string(c10::toString(promoted_type)) + 
                                        " but got " + 
                                        std::string(c10::toString(result.scalar_type())));
            }
            
            // Test explicit type promotion
            torch::Tensor tensor1_promoted = tensor1.to(promoted_type);
            torch::Tensor tensor2_promoted = tensor2.to(promoted_type);
            
            // Perform operations with explicitly promoted tensors
            torch::Tensor result2 = tensor1_promoted + tensor2_promoted;
            torch::Tensor result3 = tensor1_promoted * tensor2_promoted;
            torch::Tensor result4 = torch::matmul(tensor1_promoted, tensor2_promoted);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
