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
        
        // Need at least a few bytes to create a tensor and extract an integer
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract an integer value from the remaining data
        int64_t value = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Test torch.sym_int with different inputs
        
        // 1. Create a symbolic integer from a regular integer
        c10::SymInt sym_int1 = c10::SymInt(value);
        
        // 2. Create a symbolic integer from a tensor (if tensor is a scalar or has one element)
        if (tensor.numel() == 1) {
            // Try to convert tensor to a symbolic integer
            // This will work for integer tensor types
            if (tensor.scalar_type() == torch::kInt || 
                tensor.scalar_type() == torch::kLong ||
                tensor.scalar_type() == torch::kShort ||
                tensor.scalar_type() == torch::kByte) {
                c10::SymInt sym_int2 = c10::SymInt(tensor.item<int64_t>());
            }
        }
        
        // 3. Test symbolic integer operations
        c10::SymInt sym_int3 = c10::SymInt(5);
        
        // Addition
        c10::SymInt result1 = sym_int1 + sym_int3;
        
        // Subtraction
        c10::SymInt result2 = sym_int1 - sym_int3;
        
        // Multiplication
        c10::SymInt result3 = sym_int1 * sym_int3;
        
        // Division (be careful with zero)
        if (sym_int3.expect_int() != 0) {
            c10::SymInt result4 = sym_int1 / sym_int3;
        }
        
        // 4. Test creating a tensor with symbolic shape
        if (value > 0 && value < 1000) {  // Reasonable size to avoid OOM
            c10::SymInt sym_shape = c10::SymInt(value);
            torch::Tensor t = torch::zeros({sym_shape.expect_int()});
        }
        
        // 5. Test comparison operations
        bool eq = sym_int1 == sym_int3;
        bool neq = sym_int1 != sym_int3;
        bool lt = sym_int1 < sym_int3;
        bool gt = sym_int1 > sym_int3;
        bool lte = sym_int1 <= sym_int3;
        bool gte = sym_int1 >= sym_int3;
        
        // 6. Test conversion back to concrete int
        int64_t concrete = sym_int1.expect_int();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
