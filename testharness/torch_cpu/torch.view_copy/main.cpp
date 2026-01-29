#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        std::vector<int64_t> new_shape;
        int64_t total_elements = input_tensor.numel();
        
        uint8_t new_rank = 0;
        if (offset < Size) {
            new_rank = fuzzer_utils::parseRank(Data[offset++]);
        } else {
            new_rank = 1;
        }
        
        if (new_rank > 0) {
            if (offset < Size) {
                new_shape = fuzzer_utils::parseShape(Data, offset, Size, new_rank);
            } else {
                new_shape.push_back(total_elements);
            }
        }
        
        // Apply view_copy operation - may throw if shapes are incompatible
        torch::Tensor result;
        try {
            if (new_shape.empty()) {
                result = torch::view_copy(input_tensor, {});
            } else {
                result = torch::view_copy(input_tensor, new_shape);
            }
        } catch (const c10::Error&) {
            // Shape incompatibility is expected for fuzzed inputs
        }
        
        // Test edge cases with additional view_copy operations
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            switch (edge_case % 5) {
                case 0:
                    // Flatten the tensor
                    try {
                        result = torch::view_copy(input_tensor, {-1});
                    } catch (const c10::Error&) {}
                    break;
                case 1:
                    // Reshape with -1 dimension (auto-inferred)
                    if (input_tensor.dim() > 1 && input_tensor.numel() > 0) {
                        try {
                            std::vector<int64_t> inferred_shape;
                            inferred_shape.push_back(input_tensor.size(0));
                            inferred_shape.push_back(-1);
                            result = torch::view_copy(input_tensor, inferred_shape);
                        } catch (const c10::Error&) {}
                    }
                    break;
                case 2:
                    // Add a dimension of size 1
                    try {
                        std::vector<int64_t> expanded_shape = input_tensor.sizes().vec();
                        expanded_shape.push_back(1);
                        result = torch::view_copy(input_tensor, expanded_shape);
                    } catch (const c10::Error&) {}
                    break;
                case 3:
                    // Remove a dimension of size 1
                    if (input_tensor.dim() > 1) {
                        for (int64_t i = 0; i < input_tensor.dim(); i++) {
                            if (input_tensor.size(i) == 1) {
                                try {
                                    std::vector<int64_t> squeezed_shape;
                                    for (int64_t j = 0; j < input_tensor.dim(); j++) {
                                        if (j != i) {
                                            squeezed_shape.push_back(input_tensor.size(j));
                                        }
                                    }
                                    result = torch::view_copy(input_tensor, squeezed_shape);
                                } catch (const c10::Error&) {}
                                break;
                            }
                        }
                    }
                    break;
                case 4:
                    // Random valid shape with same number of elements
                    if (input_tensor.numel() > 1) {
                        try {
                            std::vector<int64_t> random_shape;
                            int64_t remaining = input_tensor.numel();
                            
                            while (remaining > 1 && offset < Size && random_shape.size() < 4) {
                                uint8_t factor_byte = Data[offset++];
                                int64_t factor = (factor_byte % remaining) + 1;
                                if (factor > 1 && remaining % factor == 0) {
                                    random_shape.push_back(factor);
                                    remaining /= factor;
                                }
                            }
                            
                            if (remaining > 0) {
                                random_shape.push_back(remaining);
                            }
                            
                            result = torch::view_copy(input_tensor, random_shape);
                        } catch (const c10::Error&) {}
                    }
                    break;
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