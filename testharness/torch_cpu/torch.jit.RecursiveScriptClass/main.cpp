#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

// Define a simple class to be used with RecursiveScriptClass
struct MyScriptClass : public torch::jit::CustomClassHolder {
    int64_t value;
    torch::Tensor tensor;

    MyScriptClass(int64_t val, torch::Tensor t) : value(val), tensor(t) {}
    
    int64_t getValue() const {
        return value;
    }
    
    torch::Tensor getTensor() const {
        return tensor;
    }
    
    void setValue(int64_t val) {
        value = val;
    }
    
    void setTensor(torch::Tensor t) {
        tensor = t;
    }
};

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get an integer value from the fuzzer data
        int64_t value = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Register the custom class with torch::jit
        static bool class_registered = false;
        if (!class_registered) {
            torch::class_<MyScriptClass>("__torch__", "MyScriptClass")
                .def(torch::init<int64_t, torch::Tensor>())
                .def("getValue", &MyScriptClass::getValue)
                .def("getTensor", &MyScriptClass::getTensor)
                .def("setValue", &MyScriptClass::setValue)
                .def("setTensor", &MyScriptClass::setTensor);
            class_registered = true;
        }
        
        // Create an instance of the custom class
        auto obj = c10::make_intrusive<MyScriptClass>(value, input_tensor);
        
        // Test RecursiveScriptClass functionality
        
        // 1. Test serialization/deserialization
        std::stringstream ss;
        torch::jit::Module module("test_module");
        module.register_attribute("obj", torch::getCustomClassType<MyScriptClass>(), obj);
        module.save(ss);
        
        // 2. Load the module back
        torch::jit::Module loaded_module = torch::jit::load(ss);
        
        // 3. Get the object back
        c10::intrusive_ptr<MyScriptClass> loaded_obj = loaded_module.attr("obj").toCustomClass<MyScriptClass>();
        
        // 4. Verify the object's properties
        if (loaded_obj->getValue() != value) {
            throw std::runtime_error("Value mismatch after serialization/deserialization");
        }
        
        // 5. Test method calls on the loaded object
        loaded_obj->setValue(value + 1);
        
        // 6. Test with modified tensor
        if (input_tensor.numel() > 0) {
            torch::Tensor modified_tensor;
            try {
                modified_tensor = input_tensor * 2;
                loaded_obj->setTensor(modified_tensor);
            } catch (const std::exception& e) {
                // Handle potential errors in tensor operations
            }
        }
        
        // 7. Test with empty tensor
        try {
            loaded_obj->setTensor(torch::empty({0}));
        } catch (const std::exception& e) {
            // Handle potential errors
        }
        
        // 8. Test with different tensor types
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            try {
                torch::Tensor type_tensor = input_tensor.to(dtype);
                loaded_obj->setTensor(type_tensor);
            } catch (const std::exception& e) {
                // Handle potential errors in type conversion
            }
        }
        
        // 9. Test with potentially problematic values
        if (offset + sizeof(int64_t) <= Size) {
            int64_t problematic_value;
            std::memcpy(&problematic_value, Data + offset, sizeof(int64_t));
            loaded_obj->setValue(problematic_value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
