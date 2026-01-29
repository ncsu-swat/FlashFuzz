#include "fuzzer_utils.h"
#include <iostream>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

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
                // Register Tensor attribute
                module.register_attribute("tensor_attr", 
                    torch::jit::TensorType::get(), 
                    torch::jit::IValue(tensor));
                break;
            }
            case 1: {
                // Register List[Tensor] attribute
                c10::List<torch::Tensor> tensor_list;
                tensor_list.push_back(tensor);
                module.register_attribute("list_attr", 
                    torch::jit::ListType::create(torch::jit::TensorType::get()), 
                    torch::jit::IValue(tensor_list));
                break;
            }
            case 2: {
                // Register Optional[Tensor] attribute (with value)
                module.register_attribute("optional_attr", 
                    torch::jit::OptionalType::create(torch::jit::TensorType::get()), 
                    torch::jit::IValue(tensor));
                break;
            }
            case 3: {
                // Register int attribute
                int64_t int_val = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&int_val, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                module.register_attribute("int_attr", 
                    torch::jit::IntType::get(), 
                    torch::jit::IValue(int_val));
                break;
            }
            case 4: {
                // Register bool attribute
                bool bool_val = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
                module.register_attribute("bool_attr", 
                    torch::jit::BoolType::get(), 
                    torch::jit::IValue(bool_val));
                break;
            }
        }

        // Try to get the annotated attribute back
        try {
            if (annotation_type % 5 == 0 && module.hasattr("tensor_attr")) {
                torch::Tensor retrieved = module.attr("tensor_attr").toTensor();
                (void)retrieved;
            } else if (annotation_type % 5 == 1 && module.hasattr("list_attr")) {
                auto retrieved = module.attr("list_attr").toTensorList();
                (void)retrieved;
            } else if (annotation_type % 5 == 2 && module.hasattr("optional_attr")) {
                auto retrieved = module.attr("optional_attr").toOptional<torch::Tensor>();
                (void)retrieved;
            } else if (annotation_type % 5 == 3 && module.hasattr("int_attr")) {
                auto retrieved = module.attr("int_attr").toInt();
                (void)retrieved;
            } else if (annotation_type % 5 == 4 && module.hasattr("bool_attr")) {
                auto retrieved = module.attr("bool_attr").toBool();
                (void)retrieved;
            }
        } catch (...) {
            // Silently handle retrieval failures
        }

        // Try to save and load the module with annotations
        try {
            std::stringstream ss;
            module.save(ss);
            torch::jit::Module loaded_module = torch::jit::load(ss);

            // Try to access the attributes from the loaded module
            if (annotation_type % 5 == 0 && loaded_module.hasattr("tensor_attr")) {
                torch::Tensor retrieved = loaded_module.attr("tensor_attr").toTensor();
                (void)retrieved;
            } else if (annotation_type % 5 == 1 && loaded_module.hasattr("list_attr")) {
                auto retrieved = loaded_module.attr("list_attr").toTensorList();
                (void)retrieved;
            } else if (annotation_type % 5 == 2 && loaded_module.hasattr("optional_attr")) {
                auto retrieved = loaded_module.attr("optional_attr").toOptional<torch::Tensor>();
                (void)retrieved;
            } else if (annotation_type % 5 == 3 && loaded_module.hasattr("int_attr")) {
                auto retrieved = loaded_module.attr("int_attr").toInt();
                (void)retrieved;
            } else if (annotation_type % 5 == 4 && loaded_module.hasattr("bool_attr")) {
                auto retrieved = loaded_module.attr("bool_attr").toBool();
                (void)retrieved;
            }
        } catch (...) {
            // Silently handle save/load failures - expected for some configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}