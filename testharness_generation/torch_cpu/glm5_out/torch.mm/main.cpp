#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        size_t offset = 0;

        // Parse data type selector for first tensor
        if (offset >= Size) return 0;
        uint8_t dtype_selector = Data[offset++];

        // Parse rank for first tensor (constrain to 2D for mm)
        if (offset >= Size) return 0;
        uint8_t rank1_byte = Data[offset++];
        uint8_t rank1 = 2 + (rank1_byte % 1); // Always rank 2 for mm

        // Parse shape for first tensor
        auto shape1 = fuzzer_utils::parseShape(Data, offset, Size, rank1);
        if (shape1.size() != 2) {
            shape1 = {2, 3}; // Default shape if parsing fails
        }

        // Ensure dimensions are within reasonable bounds
        for (auto& dim : shape1) {
            if (dim <= 0) dim = 1;
            if (dim > 16) dim = 16;
        }

        // Parse data type for first tensor
        auto dtype1 = fuzzer_utils::parseDataType(dtype_selector);

        // Create first tensor
        auto options1 = torch::TensorOptions().dtype(dtype1);
        torch::Tensor input1;
        if (shape1[0] > 0 && shape1[1] > 0) {
            input1 = torch::randn(shape1, options1);
        } else {
            input1 = torch::empty(shape1, options1);
        }

        // Parse data type selector for second tensor
        if (offset >= Size) return 0;
        uint8_t dtype2_selector = Data[offset++];

        // Parse rank for second tensor (always 2D)
        if (offset >= Size) return 0;
        uint8_t rank2_byte = Data[offset++];
        uint8_t rank2 = 2 + (rank2_byte % 1); // Always rank 2

        // Parse shape for second tensor
        auto shape2 = fuzzer_utils::parseShape(Data, offset, Size, rank2);
        if (shape2.size() != 2) {
            shape2 = {shape1[1], 2}; // Compatible shape
        }

        // Ensure dimensions are within reasonable bounds
        for (auto& dim : shape2) {
            if (dim <= 0) dim = 1;
            if (dim > 16) dim = 16;
        }

        // Parse data type for second tensor
        auto dtype2 = fuzzer_utils::parseDataType(dtype2_selector);

        // Create second tensor
        auto options2 = torch::TensorOptions().dtype(dtype2);
        torch::Tensor input2;
        if (shape2[0] > 0 && shape2[1] > 0) {
            input2 = torch::randn(shape2, options2);
        } else {
            input2 = torch::empty(shape2, options2);
        }

        // Parse out_dtype flag
        torch::optional<torch::ScalarType> out_dtype = torch::nullopt;
        if (offset < Size) {
            uint8_t out_dtype_flag = Data[offset++];
            if (out_dtype_flag % 4 == 0) {
                out_dtype = torch::kFloat32;
            } else if (out_dtype_flag % 4 == 1) {
                out_dtype = torch::kFloat16;
            } else if (out_dtype_flag % 4 == 2) {
                out_dtype = torch::kBFloat16;
            }
            // else nullopt
        }

        // Test edge cases: empty tensors, 1x1, mismatched inner dims
        // Override shapes occasionally based on fuzz input
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            switch (edge_case % 8) {
                case 0: // Normal case
                    break;
                case 1: // 1x1 matrices
                    input1 = torch::ones({1, 1}, options1);
                    input2 = torch::ones({1, 1}, options2);
                    break;
                case 2: // Empty first dimension
                    input1 = torch::empty({0, 3}, options1);
                    input2 = torch::empty({3, 2}, options2);
                    break;
                case 3: // Empty second dimension
                    input1 = torch::empty({2, 0}, options1);
                    input2 = torch::empty({0, 3}, options2);
                    break;
                case 4: // Empty result dimension
                    input1 = torch::empty({2, 3}, options1);
                    input2 = torch::empty({3, 0}, options2);
                    break;
                case 5: // Large inner dimension
                    input1 = torch::randn({2, 100}, options1);
                    input2 = torch::randn({100, 3}, options2);
                    break;
                case 6: // Single row
                    input1 = torch::randn({1, 5}, options1);
                    input2 = torch::randn({5, 1}, options2);
                    break;
                case 7: // Single column
                    input1 = torch::randn({5, 1}, options1);
                    input2 = torch::randn({1, 5}, options2);
                    break;
            }
        }

        // Perform matrix multiplication
        torch::Tensor result = torch::mm(input1, input2);

        // Verify result shape: (n x m) @ (m x p) = (n x p)
        if (input1.size(0) > 0 && input1.size(1) > 0 && 
            input2.size(0) > 0 && input2.size(1) > 0) {
            auto expected_shape = torch::IntArrayRef({input1.size(0), input2.size(1)});
            if (result.sizes() != expected_shape) {
                std::cerr << "Shape mismatch: expected " << expected_shape 
                          << ", got " << result.sizes() << std::endl;
            }
        }

    } catch (const c10::Error &e) {
        // Catch PyTorch errors specifically - these are expected for invalid inputs
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}