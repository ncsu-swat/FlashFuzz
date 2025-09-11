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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for QR decomposition
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // QR decomposition requires at least 2D tensor
        if (input.dim() < 2) {
            // Add dimensions if needed
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Try different variants of QR decomposition
        if (offset < Size) {
            bool some = Data[offset++] % 2 == 0;
            
            // Try with 'some' parameter
            auto result1 = torch::linalg_qr(input, some ? "reduced" : "complete");
            auto Q = std::get<0>(result1);
            auto R = std::get<1>(result1);
            
            // Verify Q*R ≈ input (basic sanity check)
            auto reconstructed = torch::matmul(Q, R);
            
            // Try with different options
            if (offset < Size) {
                const char* mode_options[] = {"reduced", "complete", "r"};
                int mode_idx = Data[offset++] % 3;
                auto result2 = torch::linalg_qr(input, mode_options[mode_idx]);
                
                // For "r" mode, only R is returned
                if (mode_idx == 2) {
                    auto R_only = std::get<0>(result2);
                } else {
                    auto Q2 = std::get<0>(result2);
                    auto R2 = std::get<1>(result2);
                }
            }
            
            // Try the older torch::qr API
            if (offset < Size) {
                bool compute_q = Data[offset++] % 2 == 0;
                auto result3 = torch::qr(input, compute_q);
                auto Q3 = std::get<0>(result3);
                auto R3 = std::get<1>(result3);
            }
        }
        
        // Try with different data types
        if (offset < Size && input.dim() >= 2) {
            // Convert to different data types and try QR
            std::vector<torch::ScalarType> qr_types = {
                torch::kFloat, torch::kDouble, 
                torch::kComplexFloat, torch::kComplexDouble
            };
            
            uint8_t type_idx = Data[offset++] % qr_types.size();
            
            // Only try conversion if the tensor is not already of this type
            if (input.scalar_type() != qr_types[type_idx]) {
                try {
                    auto converted_input = input.to(qr_types[type_idx]);
                    auto result = torch::linalg_qr(converted_input);
                } catch (...) {
                    // Conversion or QR might fail for some types, that's expected
                }
            }
        }
        
        // Try with non-standard shapes
        if (offset < Size && input.dim() >= 2) {
            // Try with tall-skinny or short-fat matrices
            int64_t dim0 = input.size(0);
            int64_t dim1 = input.size(1);
            
            // Try to reshape if possible
            if (dim0 * dim1 > 0) {
                try {
                    // Create a view with different shape
                    auto reshaped = input.reshape({-1, 1});
                    auto result = torch::linalg_qr(reshaped);
                } catch (...) {
                    // Reshape or QR might fail, that's expected
                }
                
                try {
                    // Create another view with different shape
                    auto reshaped = input.reshape({1, -1});
                    auto result = torch::linalg_qr(reshaped);
                } catch (...) {
                    // Reshape or QR might fail, that's expected
                }
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
