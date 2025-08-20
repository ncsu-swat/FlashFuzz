#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor A = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse p-norm parameter from the input data
        c10::optional<c10::Scalar> p_norm = c10::nullopt;
        if (offset < Size) {
            uint8_t p_selector = Data[offset++];
            
            // Select p-norm based on the input data
            switch (p_selector % 5) {
                case 0:
                    p_norm = c10::nullopt; // Default (Frobenius norm)
                    break;
                case 1:
                    p_norm = 1;
                    break;
                case 2:
                    p_norm = 2;
                    break;
                case 3:
                    p_norm = -1;
                    break;
                case 4:
                    p_norm = std::numeric_limits<double>::infinity();
                    break;
            }
        }
        
        // Apply torch.linalg.cond operation
        torch::Tensor result;
        
        // Try different variants of the operation
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            
            if (variant % 2 == 0) {
                // Variant 1: Just matrix and p-norm
                result = torch::linalg_cond(A, p_norm);
            } else {
                // Variant 2: With dtype specified
                torch::ScalarType dtype = fuzzer_utils::parseDataType(variant);
                result = torch::linalg_cond(A, p_norm).to(dtype);
            }
        } else {
            // Default variant if no more data
            result = torch::linalg_cond(A, p_norm);
        }
        
        // Access the result to ensure computation is performed
        if (result.defined()) {
            if (result.numel() > 0) {
                float value = result.item<float>();
                (void)value; // Prevent unused variable warning
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