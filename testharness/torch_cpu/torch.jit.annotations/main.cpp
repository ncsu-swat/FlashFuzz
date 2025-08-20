#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/script.h>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use with annotations
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract some bytes to determine annotation type
        uint8_t annotation_type = 0;
        if (offset < Size) {
            annotation_type = Data[offset++];
        }
        
        // Create a simple script module
        torch::jit::Module module("test_module");
        
        // Apply different types of annotations based on the extracted byte
        switch (annotation_type % 5) {
            case 0: {
                // Annotate as Tensor
                module.register_attribute("tensor_attr", torch::jit::TensorType::get(), tensor);
                break;
            }
            case 1: {
                // Annotate as List[Tensor]
                std::vector<torch::Tensor> tensor_list = {tensor};
                module.register_attribute("list_attr", torch::jit::ListType::create(torch::jit::TensorType::get()), tensor_list);
                break;
            }
            case 2: {
                // Annotate as Optional[Tensor]
                c10::optional<torch::Tensor> optional_tensor = tensor;
                module.register_attribute("optional_attr", torch::jit::OptionalType::create(torch::jit::TensorType::get()), optional_tensor);
                break;
            }
            case 3: {
                // Annotate as Dict[str, Tensor]
                c10::Dict<std::string, torch::Tensor> dict;
                dict.insert("key", tensor);
                module.register_attribute("dict_attr", 
                    torch::jit::DictType::create(torch::jit::StringType::get(), torch::jit::TensorType::get()), 
                    dict);
                break;
            }
            case 4: {
                // Annotate as Tuple[Tensor, Tensor]
                std::vector<torch::Tensor> tuple_elements = {tensor, tensor};
                std::vector<torch::jit::TypePtr> element_types = {torch::jit::TensorType::get(), torch::jit::TensorType::get()};
                module.register_attribute("tuple_attr", 
                    torch::jit::TupleType::create(element_types), 
                    tuple_elements);
                break;
            }
        }
        
        // Try to get the annotated attribute back
        if (annotation_type % 5 == 0) {
            torch::Tensor retrieved = module.attr("tensor_attr").toTensor();
        } else if (annotation_type % 5 == 1) {
            auto retrieved = module.attr("list_attr").toTensorList();
        } else if (annotation_type % 5 == 2) {
            auto retrieved = module.attr("optional_attr").toOptional<torch::Tensor>();
        } else if (annotation_type % 5 == 3) {
            auto retrieved = module.attr("dict_attr").toGenericDict();
        } else if (annotation_type % 5 == 4) {
            auto retrieved = module.attr("tuple_attr").toTensorVector();
        }
        
        // Try to save and load the module with annotations
        std::stringstream ss;
        module.save(ss);
        torch::jit::Module loaded_module = torch::jit::load(ss);
        
        // Try to access the attributes from the loaded module
        if (annotation_type % 5 == 0 && loaded_module.hasattr("tensor_attr")) {
            torch::Tensor retrieved = loaded_module.attr("tensor_attr").toTensor();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}