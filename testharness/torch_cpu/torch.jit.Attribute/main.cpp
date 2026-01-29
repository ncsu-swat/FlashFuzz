#include "fuzzer_utils.h"
#include <iostream>
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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some bytes for attribute name and type
        uint8_t name_selector = (offset < Size) ? Data[offset++] : 0;
        uint8_t type_selector = (offset < Size) ? Data[offset++] : 0;
        
        // Generate attribute name
        std::string attr_name = "attr_" + std::to_string(name_selector);
        
        // Create a module to hold the attribute
        torch::jit::Module module("test_module");
        
        // Register the attribute with different types based on type_selector
        switch (type_selector % 5) {
            case 0: {
                // Register tensor attribute
                module.register_attribute(attr_name, c10::TensorType::get(), torch::jit::IValue(tensor));
                break;
            }
            case 1: {
                // Register int attribute
                int64_t int_value = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&int_value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                module.register_attribute(attr_name, c10::IntType::get(), torch::jit::IValue(int_value));
                break;
            }
            case 2: {
                // Register float attribute
                double float_value = 0.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&float_value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    // Sanitize NaN/Inf values
                    if (std::isnan(float_value) || std::isinf(float_value)) {
                        float_value = 0.0;
                    }
                }
                module.register_attribute(attr_name, c10::FloatType::get(), torch::jit::IValue(float_value));
                break;
            }
            case 3: {
                // Register string attribute
                std::string str_value = "test_string_";
                if (offset < Size) {
                    str_value += std::to_string(Data[offset++]);
                }
                module.register_attribute(attr_name, c10::StringType::get(), torch::jit::IValue(str_value));
                break;
            }
            case 4: {
                // Register bool attribute
                bool bool_value = false;
                if (offset < Size) {
                    bool_value = (Data[offset++] % 2 == 0);
                }
                module.register_attribute(attr_name, c10::BoolType::get(), torch::jit::IValue(bool_value));
                break;
            }
        }
        
        // Try to retrieve the attribute
        if (module.hasattr(attr_name)) {
            switch (type_selector % 5) {
                case 0: {
                    torch::Tensor retrieved = module.attr(attr_name).toTensor();
                    (void)retrieved;
                    break;
                }
                case 1: {
                    int64_t retrieved = module.attr(attr_name).toInt();
                    (void)retrieved;
                    break;
                }
                case 2: {
                    double retrieved = module.attr(attr_name).toDouble();
                    (void)retrieved;
                    break;
                }
                case 3: {
                    std::string retrieved = module.attr(attr_name).toStringRef();
                    (void)retrieved;
                    break;
                }
                case 4: {
                    bool retrieved = module.attr(attr_name).toBool();
                    (void)retrieved;
                    break;
                }
            }
        }
        
        // Try to access a non-existent attribute (should throw an exception)
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                auto nonexistent = module.attr("nonexistent_attr");
                (void)nonexistent;
            } catch (const c10::Error&) {
                // Expected exception - catch silently
            }
        }
        
        // Try to modify an attribute
        if (offset < Size && Data[offset] % 3 == 0) {
            if (module.hasattr(attr_name)) {
                try {
                    switch (type_selector % 5) {
                        case 0: {
                            torch::Tensor new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                            module.setattr(attr_name, torch::jit::IValue(new_tensor));
                            break;
                        }
                        case 1: {
                            int64_t new_int = 42;
                            module.setattr(attr_name, torch::jit::IValue(new_int));
                            break;
                        }
                        case 2: {
                            double new_float = 3.14159;
                            module.setattr(attr_name, torch::jit::IValue(new_float));
                            break;
                        }
                        case 3: {
                            std::string new_str = "modified_string";
                            module.setattr(attr_name, torch::jit::IValue(new_str));
                            break;
                        }
                        case 4: {
                            bool new_bool = true;
                            module.setattr(attr_name, torch::jit::IValue(new_bool));
                            break;
                        }
                    }
                } catch (const c10::Error&) {
                    // Type mismatch or other expected errors - catch silently
                }
            }
        }
        
        // Try to remove an attribute
        if (offset < Size && Data[offset] % 5 == 0) {
            if (module.hasattr(attr_name)) {
                try {
                    module._ivalue()->unsafeRemoveAttr(attr_name);
                } catch (const c10::Error&) {
                    // Expected error for certain attributes - catch silently
                }
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