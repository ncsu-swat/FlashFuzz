#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a module with a Final attribute
        torch::jit::Module module("TestModule");
        
        // Register the tensor as a Final attribute
        module.register_attribute("final_tensor", tensor.type(), tensor, true);
        
        // Try to access the Final attribute
        torch::Tensor retrieved_tensor = module.attr("final_tensor").toTensor();
        
        // Try to modify the Final attribute (should throw an exception)
        try {
            module.setattr("final_tensor", torch::ones_like(tensor));
        } catch (const c10::Error& e) {
            // Expected behavior - Final attributes cannot be modified
        }
        
        // Test with different attribute names
        if (offset < Size) {
            std::string attr_name = "attr_";
            attr_name += std::to_string(Data[offset++] % 100);
            module.register_attribute(attr_name, tensor.type(), tensor, true);
            torch::Tensor retrieved = module.attr(attr_name).toTensor();
        }
        
        // Test with different tensor types
        if (offset + 1 < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            module.register_attribute("another_final", another_tensor.type(), another_tensor, true);
            torch::Tensor retrieved = module.attr("another_final").toTensor();
        }
        
        // Test with scalar values
        if (offset < Size) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset++]);
            module.register_attribute("final_scalar", c10::IntType::get(), scalar_value, true);
            int64_t retrieved_scalar = module.attr("final_scalar").toInt();
        }
        
        // Test with boolean values
        if (offset < Size) {
            bool bool_value = Data[offset++] % 2 == 0;
            module.register_attribute("final_bool", c10::BoolType::get(), bool_value, true);
            bool retrieved_bool = module.attr("final_bool").toBool();
        }
        
        // Test with string values
        if (offset + 1 < Size) {
            std::string str_value = "test_string_";
            str_value += std::to_string(Data[offset++]);
            module.register_attribute("final_string", c10::StringType::get(), str_value, true);
            std::string retrieved_str = module.attr("final_string").toStringRef();
        }
        
        // Test with double values
        if (offset < Size) {
            double double_value = static_cast<double>(Data[offset++]);
            module.register_attribute("final_double", c10::FloatType::get(), double_value, true);
            double retrieved_double = module.attr("final_double").toDouble();
        }
        
        // Test serialization and deserialization
        std::stringstream ss;
        module.save(ss);
        torch::jit::Module loaded_module = torch::jit::load(ss);
        
        // Verify the loaded module has the same attributes
        torch::Tensor loaded_tensor = loaded_module.attr("final_tensor").toTensor();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
