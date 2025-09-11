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
                module.register_attribute(attr_name, c10::TensorType::get(), tensor);
                break;
            }
            case 1: {
                // Register int attribute
                int64_t int_value = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&int_value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                module.register_attribute(attr_name, c10::IntType::get(), int_value);
                break;
            }
            case 2: {
                // Register float attribute
                double float_value = 0.0;
                if (offset + sizeof(double) <= Size) {
                    std::memcpy(&float_value, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                module.register_attribute(attr_name, c10::FloatType::get(), float_value);
                break;
            }
            case 3: {
                // Register string attribute
                std::string str_value = "test_string_";
                if (offset < Size) {
                    str_value += std::to_string(Data[offset++]);
                }
                module.register_attribute(attr_name, c10::StringType::get(), str_value);
                break;
            }
            case 4: {
                // Register bool attribute
                bool bool_value = false;
                if (offset < Size) {
                    bool_value = (Data[offset++] % 2 == 0);
                }
                module.register_attribute(attr_name, c10::BoolType::get(), bool_value);
                break;
            }
        }
        
        // Try to retrieve the attribute
        if (module.hasattr(attr_name)) {
            switch (type_selector % 5) {
                case 0: {
                    // Get tensor attribute
                    torch::Tensor retrieved = module.attr(attr_name).toTensor();
                    break;
                }
                case 1: {
                    // Get int attribute
                    int64_t retrieved = module.attr(attr_name).toInt();
                    break;
                }
                case 2: {
                    // Get float attribute
                    double retrieved = module.attr(attr_name).toDouble();
                    break;
                }
                case 3: {
                    // Get string attribute
                    std::string retrieved = module.attr(attr_name).toStringRef();
                    break;
                }
                case 4: {
                    // Get bool attribute
                    bool retrieved = module.attr(attr_name).toBool();
                    break;
                }
            }
        }
        
        // Try to access a non-existent attribute (should throw an exception)
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                auto nonexistent = module.attr("nonexistent_attr");
            } catch (const c10::Error&) {
                // Expected exception
            }
        }
        
        // Try to modify an attribute
        if (offset < Size && Data[offset] % 3 == 0) {
            if (module.hasattr(attr_name)) {
                switch (type_selector % 5) {
                    case 0: {
                        // Create a new tensor with different shape/values
                        torch::Tensor new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                        module.setattr(attr_name, new_tensor);
                        break;
                    }
                    case 1: {
                        // Modify int attribute
                        int64_t new_int = 42;
                        module.setattr(attr_name, new_int);
                        break;
                    }
                    case 2: {
                        // Modify float attribute
                        double new_float = 3.14159;
                        module.setattr(attr_name, new_float);
                        break;
                    }
                    case 3: {
                        // Modify string attribute
                        std::string new_str = "modified_string";
                        module.setattr(attr_name, new_str);
                        break;
                    }
                    case 4: {
                        // Modify bool attribute
                        bool new_bool = true;
                        module.setattr(attr_name, new_bool);
                        break;
                    }
                }
            }
        }
        
        // Try to remove an attribute
        if (offset < Size && Data[offset] % 5 == 0) {
            if (module.hasattr(attr_name)) {
                module._ivalue()->unsafeRemoveAttr(attr_name);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
