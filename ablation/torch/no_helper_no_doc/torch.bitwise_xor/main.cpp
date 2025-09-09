#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions and properties
        auto dims1 = parse_tensor_dims(Data, Size, offset);
        if (dims1.empty()) return 0;
        
        auto dims2 = parse_tensor_dims(Data, Size, offset);
        if (dims2.empty()) return 0;

        // Parse dtype - bitwise_xor works with integer and boolean types
        auto dtype = parse_dtype_integer_and_bool(Data, Size, offset);
        if (offset >= Size) return 0;

        // Parse device
        auto device = parse_device(Data, Size, offset);
        if (offset >= Size) return 0;

        // Create first tensor
        torch::Tensor tensor1;
        try {
            tensor1 = create_tensor(dims1, dtype, device);
        } catch (...) {
            return 0;
        }

        // Create second tensor - test both tensor-tensor and tensor-scalar operations
        bool use_scalar = (offset < Size) && (Data[offset++] % 2 == 0);
        
        if (use_scalar) {
            // Test tensor-scalar bitwise XOR
            int64_t scalar_val = 0;
            if (offset + sizeof(int64_t) <= Size) {
                scalar_val = *reinterpret_cast<const int64_t*>(Data + offset);
                offset += sizeof(int64_t);
            }
            
            // Test different scalar XOR operations
            auto result1 = torch::bitwise_xor(tensor1, scalar_val);
            auto result2 = tensor1.bitwise_xor(scalar_val);
            auto result3 = tensor1 ^ scalar_val;
            
            // Test in-place operations
            auto tensor1_copy = tensor1.clone();
            tensor1_copy.bitwise_xor_(scalar_val);
            auto tensor1_copy2 = tensor1.clone();
            tensor1_copy2 ^= scalar_val;
            
        } else {
            // Test tensor-tensor bitwise XOR
            torch::Tensor tensor2;
            try {
                tensor2 = create_tensor(dims2, dtype, device);
            } catch (...) {
                return 0;
            }

            // Test different tensor XOR operations
            auto result1 = torch::bitwise_xor(tensor1, tensor2);
            auto result2 = tensor1.bitwise_xor(tensor2);
            auto result3 = tensor1 ^ tensor2;
            
            // Test in-place operations
            auto tensor1_copy = tensor1.clone();
            tensor1_copy.bitwise_xor_(tensor2);
            auto tensor1_copy2 = tensor1.clone();
            tensor1_copy2 ^= tensor2;
            
            // Test broadcasting scenarios
            if (tensor1.numel() > 0 && tensor2.numel() > 0) {
                try {
                    // Create tensors with different but broadcastable shapes
                    auto small_tensor = torch::ones({1}, dtype).to(device);
                    auto broadcast_result1 = torch::bitwise_xor(tensor1, small_tensor);
                    auto broadcast_result2 = torch::bitwise_xor(small_tensor, tensor1);
                } catch (...) {
                    // Broadcasting might fail, which is acceptable
                }
            }
        }

        // Test edge cases with specific values
        if (tensor1.numel() > 0) {
            try {
                // Test XOR with zero
                auto zero_result = torch::bitwise_xor(tensor1, 0);
                
                // Test XOR with all ones (if integer type)
                if (dtype != torch::kBool) {
                    auto ones_result = torch::bitwise_xor(tensor1, -1);
                }
                
                // Test self XOR (should result in zeros)
                auto self_xor = torch::bitwise_xor(tensor1, tensor1);
                
            } catch (...) {
                // Some operations might fail for certain dtypes/values
            }
        }

        // Test with empty tensors
        try {
            auto empty_tensor = torch::empty({0}, dtype).to(device);
            auto empty_result = torch::bitwise_xor(empty_tensor, empty_tensor);
        } catch (...) {
            // Empty tensor operations might have edge cases
        }

        // Test output tensor variant if we have enough data
        if (offset < Size && tensor1.numel() > 0) {
            try {
                auto output_tensor = torch::empty_like(tensor1);
                torch::bitwise_xor_out(output_tensor, tensor1, tensor1);
            } catch (...) {
                // Output variant might fail in some cases
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