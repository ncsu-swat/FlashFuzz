#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
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
        
        // Need at least 4 bytes for tensor metadata
        if (Size < 4) {
            return 0;
        }
        
        // Create two tensors to test result_type
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch::result_type with two tensors
        auto result_type1 = torch::result_type(tensor1, tensor2);
        
        // Test with Scalar values
        if (offset + sizeof(double) <= Size) {
            double scalar_val;
            std::memcpy(&scalar_val, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle NaN/Inf for cleaner testing
            if (std::isnan(scalar_val) || std::isinf(scalar_val)) {
                scalar_val = 1.0;
            }
            
            torch::Scalar scalar1(scalar_val);
            
            // Test result_type with tensor and Scalar
            auto result_type2 = torch::result_type(tensor1, scalar1);
            
            // Test result_type with Scalar and tensor
            auto result_type3 = torch::result_type(scalar1, tensor2);
            
            // Test result_type with two Scalars
            if (offset + sizeof(double) <= Size) {
                double scalar_val2;
                std::memcpy(&scalar_val2, Data + offset, sizeof(double));
                offset += sizeof(double);
                
                if (std::isnan(scalar_val2) || std::isinf(scalar_val2)) {
                    scalar_val2 = 2.0;
                }
                
                torch::Scalar scalar2(scalar_val2);
                auto result_type4 = torch::result_type(scalar1, scalar2);
            }
        }
        
        // Test with integer Scalar
        if (offset + sizeof(int64_t) <= Size) {
            int64_t int_val;
            std::memcpy(&int_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            torch::Scalar int_scalar(int_val);
            
            // Test result_type with tensor and int Scalar
            auto result_type5 = torch::result_type(tensor1, int_scalar);
            
            // Test result_type with int Scalar and tensor
            auto result_type6 = torch::result_type(int_scalar, tensor2);
        }
        
        // Test with boolean Scalar
        if (offset < Size) {
            bool bool_val = Data[offset++] % 2 == 0;
            torch::Scalar bool_scalar(bool_val);
            
            // Test result_type with tensor and bool Scalar
            auto result_type7 = torch::result_type(tensor1, bool_scalar);
            
            // Test result_type with bool Scalar and tensor
            auto result_type8 = torch::result_type(bool_scalar, tensor2);
        }
        
        // Test with tensors of different dtypes explicitly
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 6;
            torch::ScalarType dtype;
            switch (dtype_choice) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kInt16; break;
                default: dtype = torch::kFloat32; break;
            }
            
            torch::Tensor typed_tensor = tensor1.to(dtype);
            auto result_type9 = torch::result_type(typed_tensor, tensor2);
        }
        
        // Test with complex tensors
        try {
            torch::Tensor complex_tensor = torch::complex(tensor1.to(torch::kFloat32), tensor2.to(torch::kFloat32));
            auto result_type10 = torch::result_type(complex_tensor, tensor1);
        } catch (...) {
            // Complex tensor creation may fail for some shapes, ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}