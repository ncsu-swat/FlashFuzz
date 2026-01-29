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
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Need at least 1 byte for rank
        if (offset >= Size) {
            return 0;
        }
        
        // Parse rank for target shape
        uint8_t target_rank_byte = Data[offset++];
        uint8_t target_rank = fuzzer_utils::parseRank(target_rank_byte);
        
        // Ensure target rank is at least as large as input rank for valid broadcasting
        if (target_rank < input_tensor.dim()) {
            target_rank = static_cast<uint8_t>(input_tensor.dim());
        }
        
        // Parse target shape
        std::vector<int64_t> target_shape;
        if (offset < Size) {
            target_shape = fuzzer_utils::parseShape(Data, offset, Size, target_rank);
        } else {
            // If we don't have enough data, create a simple shape
            for (uint8_t i = 0; i < target_rank; ++i) {
                target_shape.push_back(1 + (i % 5));
            }
        }
        
        // Adjust target shape to be compatible with input tensor for valid broadcasting
        // Broadcasting rules: dimensions must either match or one of them must be 1
        int64_t input_dim = input_tensor.dim();
        int64_t target_dim = static_cast<int64_t>(target_shape.size());
        
        for (int64_t i = 0; i < input_dim; ++i) {
            int64_t input_idx = input_dim - 1 - i;
            int64_t target_idx = target_dim - 1 - i;
            
            if (target_idx >= 0) {
                int64_t input_size = input_tensor.size(input_idx);
                // Make target compatible: either match input size or keep if input is 1
                if (input_size != 1 && target_shape[target_idx] != input_size) {
                    target_shape[target_idx] = input_size;
                } else if (input_size == 1 && target_shape[target_idx] < 1) {
                    target_shape[target_idx] = 1;
                }
            }
        }
        
        // Apply broadcast_to operation with valid shape
        try {
            torch::Tensor result = torch::broadcast_to(input_tensor, target_shape);
            
            // Verify the result has the expected shape
            auto result_sizes = result.sizes();
            if (result_sizes.size() != target_shape.size()) {
                throw std::runtime_error("Result rank doesn't match target rank");
            }
        } catch (const c10::Error&) {
            // Expected for some edge cases in broadcasting
        }
        
        // Try edge cases to improve coverage
        
        // Test 1: Try broadcasting scalar to various shapes
        if (offset + 1 < Size) {
            try {
                torch::Tensor scalar = torch::tensor(1.0f);
                uint8_t edge_rank = fuzzer_utils::parseRank(Data[offset++]);
                std::vector<int64_t> scalar_target;
                for (uint8_t i = 0; i < edge_rank && i < 4; ++i) {
                    scalar_target.push_back(1 + (offset < Size ? Data[offset++] % 10 : i));
                }
                if (!scalar_target.empty()) {
                    torch::Tensor broadcasted_scalar = torch::broadcast_to(scalar, scalar_target);
                }
            } catch (const c10::Error&) {
                // Expected for invalid shapes
            }
        }
        
        // Test 2: Try broadcasting with shape containing zeros
        if (!target_shape.empty()) {
            try {
                std::vector<int64_t> zero_dim_shape = target_shape;
                zero_dim_shape[0] = 0;
                torch::Tensor empty_input = torch::empty({0});
                torch::Tensor zero_result = torch::broadcast_to(empty_input, zero_dim_shape);
            } catch (const c10::Error&) {
                // Expected for invalid zero-dimension broadcasting
            }
        }
        
        // Test 3: Try broadcasting to same shape (identity)
        try {
            std::vector<int64_t> same_shape(input_tensor.sizes().begin(), input_tensor.sizes().end());
            torch::Tensor same_result = torch::broadcast_to(input_tensor, same_shape);
        } catch (const c10::Error&) {
            // Should not fail but catch anyway
        }
        
        // Test 4: Try broadcasting with expanded dimensions
        if (input_tensor.dim() > 0 && input_tensor.dim() < 5) {
            try {
                std::vector<int64_t> expanded_shape;
                expanded_shape.push_back(2); // Add leading dimension
                for (int64_t i = 0; i < input_tensor.dim(); ++i) {
                    int64_t dim_size = input_tensor.size(i);
                    expanded_shape.push_back(dim_size == 1 ? 3 : dim_size);
                }
                torch::Tensor expanded_result = torch::broadcast_to(input_tensor, expanded_shape);
            } catch (const c10::Error&) {
                // Expected if dimensions are incompatible
            }
        }
        
        // Test 5: Try with different dtypes
        if (offset < Size) {
            try {
                uint8_t dtype_byte = Data[offset++];
                torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_byte);
                torch::Tensor typed_tensor = input_tensor.to(dtype);
                torch::Tensor typed_result = torch::broadcast_to(typed_tensor, target_shape);
            } catch (const c10::Error&) {
                // Expected for some dtype conversions
            } catch (const std::exception&) {
                // Handle other conversion errors
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