#include "fuzzer_utils.h"
#include <iostream>
#include <sstream>
#include <ATen/core/jit_type.h>
#include <torch/script.h>

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
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        auto tensor_type = c10::TensorType::create(tensor);
        
        // Create a module with a Final (constant) attribute
        torch::jit::Module module("TestModule");
        
        // Register the tensor as a constant attribute (equivalent to Final in Python)
        // The last parameter 'true' means is_constant (Final)
        module.register_attribute("final_tensor", tensor_type, tensor, true);
        
        // Access the Final attribute
        torch::Tensor retrieved_tensor = module.attr("final_tensor").toTensor();
        
        // Verify tensor was retrieved correctly
        (void)retrieved_tensor.sizes();
        
        // Try to modify the Final attribute (should throw an exception)
        try {
            module.setattr("final_tensor", torch::ones_like(tensor));
            // If we get here, the modification was allowed (unexpected for Final)
        } catch (const c10::Error& e) {
            // Expected behavior - Final/constant attributes cannot be modified
        } catch (const std::exception& e) {
            // Other exceptions during modification attempt
        }
        
        // Test with different attribute names based on fuzzer data
        if (offset < Size) {
            uint8_t name_suffix = Data[offset++] % 100;
            std::string attr_name = "attr_" + std::to_string(name_suffix);
            
            try {
                torch::Tensor attr_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto attr_type = c10::TensorType::create(attr_tensor);
                module.register_attribute(attr_name, attr_type, attr_tensor, true);
                torch::Tensor retrieved = module.attr(attr_name).toTensor();
                (void)retrieved.numel();
            } catch (const std::exception& e) {
                // Attribute registration may fail for various reasons
            }
        }
        
        // Test with scalar integer values (Final)
        if (offset < Size) {
            int64_t scalar_value = static_cast<int64_t>(Data[offset++]) - 128;
            try {
                module.register_attribute("final_int", c10::IntType::get(), scalar_value, true);
                int64_t retrieved_scalar = module.attr("final_int").toInt();
                (void)retrieved_scalar;
            } catch (const std::exception& e) {
                // May fail
            }
        }
        
        // Test with boolean values (Final)
        if (offset < Size) {
            bool bool_value = (Data[offset++] % 2) == 0;
            try {
                module.register_attribute("final_bool", c10::BoolType::get(), bool_value, true);
                bool retrieved_bool = module.attr("final_bool").toBool();
                (void)retrieved_bool;
            } catch (const std::exception& e) {
                // May fail
            }
        }
        
        // Test with double values (Final)
        if (offset < Size) {
            double double_value = static_cast<double>(Data[offset++]) / 10.0;
            try {
                module.register_attribute("final_double", c10::FloatType::get(), double_value, true);
                double retrieved_double = module.attr("final_double").toDouble();
                (void)retrieved_double;
            } catch (const std::exception& e) {
                // May fail
            }
        }
        
        // Test non-constant (non-Final) attribute for comparison
        if (offset < Size) {
            torch::Tensor mutable_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto mutable_type = c10::TensorType::create(mutable_tensor);
            
            try {
                // Register as non-constant (is_constant=false)
                module.register_attribute("mutable_tensor", mutable_type, mutable_tensor, false);
                
                // This modification should succeed for non-Final attributes
                module.setattr("mutable_tensor", torch::zeros_like(mutable_tensor));
                torch::Tensor modified = module.attr("mutable_tensor").toTensor();
                (void)modified.sum();
            } catch (const std::exception& e) {
                // May fail for other reasons
            }
        }
        
        // Test serialization and deserialization of module with Final attributes
        try {
            std::stringstream ss;
            module.save(ss);
            
            // Load the module back
            ss.seekg(0);
            torch::jit::Module loaded_module = torch::jit::load(ss);
            
            // Verify the loaded module has the Final attribute preserved
            torch::Tensor loaded_tensor = loaded_module.attr("final_tensor").toTensor();
            (void)loaded_tensor.sizes();
            
            // Verify Final semantics are preserved after load
            try {
                loaded_module.setattr("final_tensor", torch::ones({1}));
            } catch (const std::exception& e) {
                // Expected - Final attribute should still be immutable
            }
        } catch (const std::exception& e) {
            // Serialization may fail for various reasons
        }
        
        // Test cloning a module with Final attributes
        try {
            torch::jit::Module cloned = module.clone();
            torch::Tensor cloned_tensor = cloned.attr("final_tensor").toTensor();
            (void)cloned_tensor.numel();
        } catch (const std::exception& e) {
            // Cloning may fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}