#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor creation
        if (Size < 16) {
            return 0;
        }

        // Extract parameters for tensor creation
        auto shape_info = extract_tensor_shape(Data, Size, offset);
        if (shape_info.empty()) {
            return 0;
        }

        // Create first tensor (dividend)
        auto tensor1 = create_tensor_from_data(Data, Size, offset, shape_info);
        if (!tensor1.defined()) {
            return 0;
        }

        // Test different remainder operations
        
        // 1. Test tensor.remainder(scalar)
        if (offset + sizeof(double) <= Size) {
            double scalar_divisor;
            std::memcpy(&scalar_divisor, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Avoid division by zero
            if (std::abs(scalar_divisor) > 1e-10) {
                auto result1 = torch::remainder(tensor1, scalar_divisor);
                auto result2 = tensor1.remainder(scalar_divisor);
                
                // Test in-place version
                auto tensor1_copy = tensor1.clone();
                tensor1_copy.remainder_(scalar_divisor);
            }
        }

        // 2. Test tensor.remainder(tensor)
        if (offset < Size) {
            // Create second tensor (divisor) with same or compatible shape
            auto tensor2_shape = shape_info;
            
            // Sometimes use broadcasting by modifying shape
            if (offset + 1 <= Size) {
                uint8_t broadcast_flag = Data[offset++];
                if (broadcast_flag % 3 == 1 && !tensor2_shape.empty()) {
                    // Make it broadcastable - set some dimensions to 1
                    for (size_t i = 0; i < tensor2_shape.size(); i += 2) {
                        tensor2_shape[i] = 1;
                    }
                } else if (broadcast_flag % 3 == 2 && tensor2_shape.size() > 1) {
                    // Reduce dimensionality
                    tensor2_shape.resize(tensor2_shape.size() - 1);
                }
            }
            
            auto tensor2 = create_tensor_from_data(Data, Size, offset, tensor2_shape);
            if (tensor2.defined()) {
                // Avoid tensors with zeros by adding small epsilon
                tensor2 = tensor2 + 1e-6;
                
                auto result3 = torch::remainder(tensor1, tensor2);
                auto result4 = tensor1.remainder(tensor2);
                
                // Test in-place version
                auto tensor1_copy2 = tensor1.clone();
                tensor1_copy2.remainder_(tensor2);
            }
        }

        // 3. Test with different dtypes
        if (offset + 1 <= Size) {
            uint8_t dtype_flag = Data[offset++];
            torch::ScalarType target_dtype;
            
            switch (dtype_flag % 6) {
                case 0: target_dtype = torch::kFloat32; break;
                case 1: target_dtype = torch::kFloat64; break;
                case 2: target_dtype = torch::kInt32; break;
                case 3: target_dtype = torch::kInt64; break;
                case 4: target_dtype = torch::kInt8; break;
                default: target_dtype = torch::kInt16; break;
            }
            
            auto typed_tensor = tensor1.to(target_dtype);
            
            if (offset + sizeof(int64_t) <= Size) {
                int64_t int_divisor;
                std::memcpy(&int_divisor, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                if (int_divisor != 0) {
                    auto result5 = torch::remainder(typed_tensor, int_divisor);
                }
            }
        }

        // 4. Test edge cases with special values
        if (offset + sizeof(float) <= Size) {
            float special_val;
            std::memcpy(&special_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Test with infinity, NaN, very large/small numbers
            auto special_tensor = torch::full_like(tensor1, special_val);
            
            if (std::isfinite(special_val) && std::abs(special_val) > 1e-10) {
                auto result6 = torch::remainder(tensor1, special_tensor);
            }
        }

        // 5. Test with empty tensors
        auto empty_tensor = torch::empty({0});
        if (empty_tensor.numel() == 0) {
            auto result7 = torch::remainder(empty_tensor, 1.0);
        }

        // 6. Test with single element tensors
        auto single_tensor = torch::randn({1});
        auto result8 = torch::remainder(single_tensor, 2.0);

        // 7. Test with negative values
        auto negative_tensor = -torch::abs(tensor1);
        if (offset + sizeof(double) <= Size) {
            double neg_divisor;
            std::memcpy(&neg_divisor, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            if (std::abs(neg_divisor) > 1e-10) {
                auto result9 = torch::remainder(negative_tensor, neg_divisor);
                auto result10 = torch::remainder(tensor1, -std::abs(neg_divisor));
            }
        }

        // 8. Test with very large tensors (if we have enough data)
        if (Size > 1000 && offset + 100 <= Size) {
            auto large_shape = std::vector<int64_t>{10, 10};
            auto large_tensor = create_tensor_from_data(Data, Size, offset, large_shape);
            if (large_tensor.defined()) {
                auto result11 = torch::remainder(large_tensor, 3.14159);
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