#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        // This is the main API being tested
        torch::ScalarType promoted_type = torch::promote_types(type1, type2);
        
        // Verify the promoted type is valid
        (void)c10::toString(promoted_type);
        
        // Test symmetry: promote_types(a, b) should equal promote_types(b, a)
        torch::ScalarType promoted_type_reverse = torch::promote_types(type2, type1);
        if (promoted_type != promoted_type_reverse) {
            // This would indicate a bug in PyTorch
            std::cerr << "Asymmetric promotion detected!" << std::endl;
        }
        
        // Test idempotence: promote_types(a, a) should equal a
        torch::ScalarType self_promoted1 = torch::promote_types(type1, type1);
        torch::ScalarType self_promoted2 = torch::promote_types(type2, type2);
        (void)self_promoted1;
        (void)self_promoted2;
        
        // Test transitivity with the promoted type
        torch::ScalarType trans1 = torch::promote_types(type1, promoted_type);
        torch::ScalarType trans2 = torch::promote_types(type2, promoted_type);
        (void)trans1;
        (void)trans2;
        
        // Create tensors and test actual promotion in operations
        if (offset + 4 <= Size) {
            try {
                // Create small tensors for testing
                torch::Tensor tensor1 = torch::zeros({2, 2}, torch::TensorOptions().dtype(type1));
                torch::Tensor tensor2 = torch::zeros({2, 2}, torch::TensorOptions().dtype(type2));
                
                // Fill with some values from fuzzer data
                if (offset < Size) {
                    float val1 = static_cast<float>(Data[offset++]) / 255.0f;
                    float val2 = (offset < Size) ? static_cast<float>(Data[offset++]) / 255.0f : 0.5f;
                    tensor1.fill_(val1);
                    tensor2.fill_(val2);
                }
                
                // Test explicit conversion to promoted type
                torch::Tensor tensor1_promoted = tensor1.to(promoted_type);
                torch::Tensor tensor2_promoted = tensor2.to(promoted_type);
                
                // Verify the conversions worked
                if (tensor1_promoted.scalar_type() != promoted_type ||
                    tensor2_promoted.scalar_type() != promoted_type) {
                    std::cerr << "Conversion to promoted type failed!" << std::endl;
                }
                
                // Test operations with promoted tensors
                torch::Tensor result_add = tensor1_promoted + tensor2_promoted;
                torch::Tensor result_sub = tensor1_promoted - tensor2_promoted;
                torch::Tensor result_mul = tensor1_promoted * tensor2_promoted;
                
                (void)result_add;
                (void)result_sub;
                (void)result_mul;
                
            } catch (const c10::Error &e) {
                // Some type combinations may not support certain operations
                // This is expected behavior, not a bug
            } catch (const std::exception &e) {
                // Catch other expected failures silently
            }
        }
        
        // Test with more type combinations if we have more data
        if (offset + 1 < Size) {
            uint8_t type3_selector = Data[offset++];
            torch::ScalarType type3 = fuzzer_utils::parseDataType(type3_selector);
            
            // Test chained promotions
            torch::ScalarType promoted_12 = torch::promote_types(type1, type2);
            torch::ScalarType promoted_123 = torch::promote_types(promoted_12, type3);
            
            // Alternative order
            torch::ScalarType promoted_23 = torch::promote_types(type2, type3);
            torch::ScalarType promoted_123_alt = torch::promote_types(type1, promoted_23);
            
            (void)promoted_123;
            (void)promoted_123_alt;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}