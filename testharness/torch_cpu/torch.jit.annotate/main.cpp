#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a byte to determine the type annotation
        uint8_t type_selector = 0;
        if (offset < Size) {
            type_selector = Data[offset++];
        }
        
        // Create IValue objects to test type annotations
        switch (type_selector % 8) {
            case 0: {
                // Create IValue from Tensor
                c10::IValue tensor_val(tensor);
                break;
            }
            case 1: {
                // Create IValue from List[Tensor]
                std::vector<torch::Tensor> tensor_list = {tensor};
                c10::IValue list_val(tensor_list);
                break;
            }
            case 2: {
                // Create IValue from Optional[Tensor]
                c10::optional<torch::Tensor> opt_tensor = tensor;
                c10::IValue opt_val(opt_tensor);
                break;
            }
            case 3: {
                // Create IValue from Dict[str, Tensor]
                c10::Dict<std::string, torch::Tensor> dict;
                dict.insert("key", tensor);
                c10::IValue dict_val(dict);
                break;
            }
            case 4: {
                // Create IValue from Tuple[Tensor, Tensor]
                std::tuple<torch::Tensor, torch::Tensor> tuple_val = std::make_tuple(tensor, tensor);
                c10::IValue tuple_ival(tuple_val);
                break;
            }
            case 5: {
                // Create IValue from int
                int64_t int_val = 42;
                c10::IValue int_ival(int_val);
                break;
            }
            case 6: {
                // Create IValue from float
                double float_val = 3.14;
                c10::IValue float_ival(float_val);
                break;
            }
            case 7: {
                // Create IValue from bool
                bool bool_val = true;
                c10::IValue bool_ival(bool_val);
                break;
            }
        }
        
        // Test type checking with IValue
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                c10::IValue tensor_val(tensor);
                bool is_tensor = tensor_val.isTensor();
                bool is_int = tensor_val.isInt();
            } catch (...) {
                // May throw, ignore
            }
        }
        
        // Test custom type creation
        if (offset < Size && Data[offset] % 3 == 0) {
            try {
                auto compilation_unit = std::make_shared<torch::jit::CompilationUnit>();
                auto custom_type = c10::ClassType::create("TestClass", compilation_unit);
            } catch (...) {
                // May throw, ignore
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