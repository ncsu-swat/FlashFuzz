#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for basic parameters
        if (Size < 16) return 0;

        // Generate input tensor with random shape and values
        auto shape = generateRandomShape(Data, Size, offset, 1, 5, 1, 10);
        auto input = generateRandomTensor(Data, Size, offset, shape);
        
        if (input.numel() == 0) return 0;

        // Test basic gradient computation (no optional parameters)
        auto result1 = torch::gradient(input);
        
        // Generate edge_order parameter (1 or 2)
        int edge_order = (consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 1) == 0) ? 1 : 2;
        
        // Test with edge_order
        auto result2 = torch::gradient(input, torch::TensorOptions(), c10::nullopt, edge_order);

        // Test with scalar spacing
        if (offset + 4 < Size) {
            double spacing_scalar = consumeFloatingPoint<double>(Data, Size, offset);
            if (std::isfinite(spacing_scalar) && spacing_scalar != 0.0) {
                auto result3 = torch::gradient(input, spacing_scalar, c10::nullopt, edge_order);
            }
        }

        // Test with specific dimensions
        if (input.dim() > 1 && offset + 1 < Size) {
            int dim_choice = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, input.dim() - 1);
            auto result4 = torch::gradient(input, torch::TensorOptions(), dim_choice, edge_order);
        }

        // Test with list of dimensions
        if (input.dim() > 2 && offset + 2 < Size) {
            std::vector<int64_t> dims;
            int num_dims = consumeIntegralInRange<uint8_t>(Data, Size, offset, 1, std::min(3, (int)input.dim()));
            for (int i = 0; i < num_dims && offset < Size; i++) {
                int dim = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, input.dim() - 1);
                if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
                    dims.push_back(dim);
                }
            }
            if (!dims.empty()) {
                auto result5 = torch::gradient(input, torch::TensorOptions(), dims, edge_order);
            }
        }

        // Test with spacing as list of scalars
        if (offset + input.dim() * 4 < Size) {
            std::vector<double> spacing_list;
            bool valid_spacing = true;
            for (int64_t i = 0; i < input.dim() && offset + 4 <= Size; i++) {
                double spacing_val = consumeFloatingPoint<double>(Data, Size, offset);
                if (!std::isfinite(spacing_val) || spacing_val == 0.0) {
                    valid_spacing = false;
                    break;
                }
                spacing_list.push_back(spacing_val);
            }
            if (valid_spacing && spacing_list.size() == input.dim()) {
                auto result6 = torch::gradient(input, spacing_list, c10::nullopt, edge_order);
            }
        }

        // Test with spacing as list of tensors
        if (input.dim() <= 3 && offset + input.dim() * 8 < Size) {
            std::vector<torch::Tensor> spacing_tensors;
            bool valid_tensor_spacing = true;
            
            for (int64_t i = 0; i < input.dim() && offset + 8 <= Size; i++) {
                int64_t tensor_size = input.size(i);
                if (tensor_size > 100) tensor_size = 100; // Limit size for fuzzing
                
                std::vector<double> coords;
                for (int64_t j = 0; j < tensor_size && offset + 4 <= Size; j++) {
                    double coord = consumeFloatingPoint<double>(Data, Size, offset);
                    if (!std::isfinite(coord)) {
                        valid_tensor_spacing = false;
                        break;
                    }
                    coords.push_back(coord);
                }
                
                if (!valid_tensor_spacing) break;
                
                if (coords.size() == tensor_size) {
                    auto coord_tensor = torch::tensor(coords, torch::kFloat64);
                    spacing_tensors.push_back(coord_tensor);
                } else {
                    valid_tensor_spacing = false;
                    break;
                }
            }
            
            if (valid_tensor_spacing && spacing_tensors.size() == input.dim()) {
                auto result7 = torch::gradient(input, spacing_tensors, c10::nullopt, edge_order);
            }
        }

        // Test edge cases with small tensors
        if (offset + 8 < Size) {
            // 1D tensor with 2 elements
            auto small_tensor = torch::randn({2});
            auto result_small = torch::gradient(small_tensor);
            
            // 1D tensor with 1 element
            auto tiny_tensor = torch::randn({1});
            auto result_tiny = torch::gradient(tiny_tensor);
        }

        // Test with different dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_choice = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, 3);
            torch::Tensor typed_input;
            
            switch (dtype_choice) {
                case 0:
                    typed_input = input.to(torch::kFloat32);
                    break;
                case 1:
                    typed_input = input.to(torch::kFloat64);
                    break;
                case 2:
                    typed_input = input.to(torch::kComplexFloat);
                    break;
                case 3:
                    typed_input = input.to(torch::kComplexDouble);
                    break;
            }
            
            auto result_typed = torch::gradient(typed_input);
        }

        // Test with negative spacing values
        if (offset + 4 < Size) {
            double neg_spacing = -std::abs(consumeFloatingPoint<double>(Data, Size, offset));
            if (std::isfinite(neg_spacing) && neg_spacing != 0.0) {
                auto result_neg = torch::gradient(input, neg_spacing);
            }
        }

        // Test combinations of parameters
        if (input.dim() > 1 && offset + 8 < Size) {
            double spacing = consumeFloatingPoint<double>(Data, Size, offset);
            int dim = consumeIntegralInRange<uint8_t>(Data, Size, offset, 0, input.dim() - 1);
            
            if (std::isfinite(spacing) && spacing != 0.0) {
                auto result_combo = torch::gradient(input, spacing, dim, edge_order);
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