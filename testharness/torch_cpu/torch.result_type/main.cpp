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
        
        // Need at least 2 bytes for each tensor's metadata
        if (Size < 4) {
            return 0;
        }
        
        // Create two tensors or scalar values to test result_type
        torch::Tensor tensor1;
        torch::Tensor tensor2;
        
        // Decide whether to use scalar or tensor for first input
        bool use_scalar1 = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        if (use_scalar1) {
            // Create a scalar value
            if (offset + sizeof(double) <= Size) {
                double scalar_value;
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                tensor1 = torch::tensor(scalar_value);
            } else {
                // Not enough data for double, use int
                int64_t scalar_value = (offset < Size) ? static_cast<int64_t>(Data[offset++]) : 1;
                tensor1 = torch::tensor(scalar_value);
            }
        } else {
            // Create a tensor
            tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Decide whether to use scalar or tensor for second input
        bool use_scalar2 = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
        
        if (use_scalar2) {
            // Create a scalar value
            if (offset + sizeof(double) <= Size) {
                double scalar_value;
                std::memcpy(&scalar_value, Data + offset, sizeof(double));
                offset += sizeof(double);
                tensor2 = torch::tensor(scalar_value);
            } else {
                // Not enough data for double, use int
                int64_t scalar_value = (offset < Size) ? static_cast<int64_t>(Data[offset++]) : 2;
                tensor2 = torch::tensor(scalar_value);
            }
        } else {
            // Create a tensor
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Test torch.result_type with the created tensors/scalars
        auto result_type = torch::result_type(tensor1, tensor2);
        
        // Test with scalar values directly
        if (offset + sizeof(double) <= Size) {
            double scalar1;
            std::memcpy(&scalar1, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Test result_type with tensor and scalar
            auto result_type2 = torch::result_type(tensor1, scalar1);
            
            // Test result_type with scalar and tensor
            auto result_type3 = torch::result_type(scalar1, tensor2);
            
            // Test result_type with two scalars
            if (offset + sizeof(double) <= Size) {
                double scalar2;
                std::memcpy(&scalar2, Data + offset, sizeof(double));
                auto result_type4 = torch::result_type(scalar1, scalar2);
            }
        }
        
        // Test with integer scalars
        if (offset + sizeof(int64_t) <= Size) {
            int64_t int_scalar;
            std::memcpy(&int_scalar, Data + offset, sizeof(int64_t));
            
            // Test result_type with tensor and int scalar
            auto result_type5 = torch::result_type(tensor1, int_scalar);
            
            // Test result_type with int scalar and tensor
            auto result_type6 = torch::result_type(int_scalar, tensor2);
        }
        
        // Test with boolean scalars
        if (offset < Size) {
            bool bool_scalar = Data[offset++] % 2 == 0;
            
            // Test result_type with tensor and bool scalar
            auto result_type7 = torch::result_type(tensor1, bool_scalar);
            
            // Test result_type with bool scalar and tensor
            auto result_type8 = torch::result_type(bool_scalar, tensor2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
