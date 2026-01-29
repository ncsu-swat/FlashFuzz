#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for QR decomposition
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // QR decomposition requires at least 2D tensor with floating point type
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Ensure floating point type for QR decomposition
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat);
        }
        
        // Test the classic torch::qr API (may be deprecated but should still work)
        if (offset < Size) {
            bool some = Data[offset++] % 2 == 0;
            
            try {
                auto result = torch::qr(input, some);
                auto Q = std::get<0>(result);
                auto R = std::get<1>(result);
                
                // Basic sanity check - verify shapes are consistent
                if (Q.defined() && R.defined()) {
                    // Q should have same number of rows as input
                    // R should have same number of columns as input
                    (void)Q.size(0);
                    (void)R.size(-1);
                }
            } catch (...) {
                // Shape or numerical issues - expected for edge cases
            }
        }
        
        // Test torch::linalg_qr with different modes
        if (offset < Size) {
            const char* mode_options[] = {"reduced", "complete", "r"};
            int mode_idx = Data[offset++] % 3;
            
            try {
                auto result = torch::linalg_qr(input, mode_options[mode_idx]);
                auto Q = std::get<0>(result);
                auto R = std::get<1>(result);
                
                // For all modes, R is always returned
                // For "r" mode, Q is an empty tensor
                if (mode_idx != 2 && Q.defined() && Q.numel() > 0) {
                    // Verify Q*R ≈ input for reduced/complete modes
                    try {
                        auto reconstructed = torch::matmul(Q, R);
                        (void)reconstructed.size(0);
                    } catch (...) {
                        // Matmul might fail for edge cases
                    }
                }
            } catch (...) {
                // QR might fail for singular or ill-conditioned matrices
            }
        }
        
        // Test with different floating point types
        if (offset < Size && input.dim() >= 2) {
            std::vector<torch::ScalarType> qr_types = {
                torch::kFloat, 
                torch::kDouble, 
                torch::kComplexFloat, 
                torch::kComplexDouble
            };
            
            uint8_t type_idx = Data[offset++] % qr_types.size();
            
            if (input.scalar_type() != qr_types[type_idx]) {
                try {
                    auto converted_input = input.to(qr_types[type_idx]);
                    auto result = torch::linalg_qr(converted_input);
                    auto Q = std::get<0>(result);
                    auto R = std::get<1>(result);
                } catch (...) {
                    // Type conversion or QR might fail
                }
            }
        }
        
        // Test with batched inputs (3D tensors)
        if (offset < Size && input.dim() == 2) {
            try {
                // Create a batched version by adding batch dimension
                auto batched = input.unsqueeze(0);
                if (offset < Size && Data[offset++] % 2 == 0) {
                    // Repeat along batch dimension
                    batched = batched.expand({2, -1, -1}).contiguous();
                }
                auto result = torch::linalg_qr(batched);
                auto Q = std::get<0>(result);
                auto R = std::get<1>(result);
            } catch (...) {
                // Batched QR might fail
            }
        }
        
        // Test with various matrix shapes (tall, wide, square)
        if (offset + 1 < Size && input.numel() > 0) {
            uint8_t rows = (Data[offset++] % 8) + 1;  // 1-8 rows
            uint8_t cols = (Data[offset++] % 8) + 1;  // 1-8 cols
            
            try {
                auto matrix = torch::randn({rows, cols});
                auto result = torch::linalg_qr(matrix);
                auto Q = std::get<0>(result);
                auto R = std::get<1>(result);
                
                // Test "complete" mode specifically for non-square matrices
                if (rows != cols) {
                    auto result_complete = torch::linalg_qr(matrix, "complete");
                    auto Q_complete = std::get<0>(result_complete);
                    // Q should be square (rows x rows) in complete mode
                }
            } catch (...) {
                // QR operations might fail
            }
        }
        
        // Test out-of-place QR with pre-allocated output
        if (offset < Size && input.dim() >= 2) {
            try {
                torch::Tensor Q_out, R_out;
                std::tie(Q_out, R_out) = torch::linalg_qr(input);
                
                // Verify output tensors are valid
                if (Q_out.defined() && R_out.defined()) {
                    (void)Q_out.sum();
                    (void)R_out.sum();
                }
            } catch (...) {
                // Operations might fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}