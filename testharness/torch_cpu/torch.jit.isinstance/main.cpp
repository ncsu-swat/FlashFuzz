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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple JIT script module
        std::string script_code = R"(
            def check_tensor_type(x):
                return torch.jit.isinstance(x, torch.Tensor)
                
            def check_int_type(x):
                return torch.jit.isinstance(x, int)
                
            def check_float_type(x):
                return torch.jit.isinstance(x, float)
                
            def check_list_type(x):
                return torch.jit.isinstance(x, list)
                
            def check_tuple_type(x):
                return torch.jit.isinstance(x, tuple)
                
            def check_dict_type(x):
                return torch.jit.isinstance(x, dict)
        )";
        
        auto compilation_unit = torch::jit::compile(script_code);
        
        // Test isinstance with tensor
        std::vector<torch::jit::IValue> inputs = {tensor};
        torch::jit::IValue result = compilation_unit->get_function("check_tensor_type")(inputs);
        bool is_tensor = result.toBool();
        
        // Test isinstance with int
        if (offset + 1 < Size) {
            int64_t int_val = static_cast<int64_t>(Data[offset++]);
            inputs = {int_val};
            result = compilation_unit->get_function("check_int_type")(inputs);
            bool is_int = result.toBool();
        }
        
        // Test isinstance with float
        if (offset + 4 <= Size) {
            float float_val;
            std::memcpy(&float_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            inputs = {float_val};
            result = compilation_unit->get_function("check_float_type")(inputs);
            bool is_float = result.toBool();
        }
        
        // Test isinstance with list
        if (offset < Size) {
            c10::impl::GenericList list_val(c10::AnyType::get());
            size_t list_size = Data[offset++] % 5; // Create a list with 0-4 elements
            
            for (size_t i = 0; i < list_size && offset < Size; i++) {
                list_val.push_back(static_cast<int64_t>(Data[offset++]));
            }
            
            inputs = {list_val};
            result = compilation_unit->get_function("check_list_type")(inputs);
            bool is_list = result.toBool();
        }
        
        // Test isinstance with tuple
        if (offset < Size) {
            std::vector<torch::jit::IValue> tuple_elements;
            size_t tuple_size = Data[offset++] % 5; // Create a tuple with 0-4 elements
            
            for (size_t i = 0; i < tuple_size && offset < Size; i++) {
                tuple_elements.push_back(static_cast<int64_t>(Data[offset++]));
            }
            
            c10::impl::GenericList list(c10::AnyType::get());
            for (const auto& elem : tuple_elements) {
                list.push_back(elem);
            }
            
            inputs = {c10::ivalue::Tuple::create(std::move(list))};
            result = compilation_unit->get_function("check_tuple_type")(inputs);
            bool is_tuple = result.toBool();
        }
        
        // Test isinstance with dict
        if (offset + 2 < Size) {
            c10::impl::GenericDict dict(c10::AnyType::get(), c10::AnyType::get());
            size_t dict_size = Data[offset++] % 3; // Create a dict with 0-2 elements
            
            for (size_t i = 0; i < dict_size && offset + 1 < Size; i++) {
                int64_t key = static_cast<int64_t>(Data[offset++]);
                int64_t value = static_cast<int64_t>(Data[offset++]);
                dict.insert(torch::jit::IValue(key), torch::jit::IValue(value));
            }
            
            inputs = {dict};
            result = compilation_unit->get_function("check_dict_type")(inputs);
            bool is_dict = result.toBool();
        }
        
        // Create a more complex test with nested types
        if (Size > 10 && offset + 5 < Size) {
            // Create a script with more complex type checking
            std::string complex_script = R"(
                def check_complex_type(x):
                    if torch.jit.isinstance(x, list):
                        for item in x:
                            if torch.jit.isinstance(item, torch.Tensor):
                                return True
                    return False
            )";
            
            auto complex_compilation_unit = torch::jit::compile(complex_script);
            
            // Create a list of tensors
            c10::impl::GenericList tensor_list(c10::AnyType::get());
            size_t list_size = Data[offset++] % 3 + 1; // 1-3 tensors
            
            for (size_t i = 0; i < list_size && offset < Size; i++) {
                torch::Tensor small_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensor_list.push_back(small_tensor);
            }
            
            std::vector<torch::jit::IValue> complex_inputs = {tensor_list};
            torch::jit::IValue complex_result = complex_compilation_unit->get_function("check_complex_type")(complex_inputs);
            bool is_complex_match = complex_result.toBool();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
