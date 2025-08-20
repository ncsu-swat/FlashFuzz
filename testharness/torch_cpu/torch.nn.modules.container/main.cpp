#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor tensor2;
        if (offset < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor2 = tensor1.clone();
        }
        
        // Create a third tensor if there's more data
        torch::Tensor tensor3;
        if (offset < Size) {
            tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor3 = tensor1.clone();
        }
        
        // Test Sequential container
        {
            torch::nn::Sequential sequential(
                torch::nn::Linear(10, 5),
                torch::nn::ReLU(),
                torch::nn::Linear(5, 1)
            );
            
            // Try to use the sequential model with our tensor
            try {
                if (tensor1.dim() > 0 && tensor1.size(0) > 0) {
                    // Reshape tensor to match input requirements if possible
                    auto batch_size = tensor1.size(0);
                    auto reshaped = tensor1.reshape({batch_size, -1});
                    
                    // If the last dimension isn't 10, resize it
                    if (reshaped.size(1) != 10) {
                        reshaped = reshaped.index({"...", torch::indexing::Slice(0, std::min(static_cast<int64_t>(10), reshaped.size(1)))});
                        if (reshaped.size(1) < 10) {
                            // Pad if needed
                            auto padding = torch::zeros({reshaped.size(0), 10 - reshaped.size(1)}, reshaped.options());
                            reshaped = torch::cat({reshaped, padding}, 1);
                        }
                    }
                    
                    auto output = sequential->forward(reshaped);
                }
            } catch (...) {
                // Catch any exceptions from the forward pass
            }
            
            // Test adding modules
            try {
                sequential->push_back("extra_linear", torch::nn::Linear(1, 2));
                sequential->push_back(torch::nn::ReLU());
            } catch (...) {
                // Catch any exceptions from adding modules
            }
            
            // Test accessing modules
            try {
                auto module = sequential[0];
                auto named_module = sequential->named_modules();
            } catch (...) {
                // Catch any exceptions from accessing modules
            }
        }
        
        // Test ModuleList container
        {
            torch::nn::ModuleList module_list;
            
            // Add modules to the list
            try {
                module_list->push_back(torch::nn::Linear(10, 5));
                module_list->push_back(torch::nn::ReLU());
                module_list->push_back(torch::nn::Linear(5, 1));
            } catch (...) {
                // Catch any exceptions from adding modules
            }
            
            // Test accessing modules
            try {
                if (module_list->size() > 0) {
                    auto module = module_list[0];
                    auto modules = module_list->modules();
                }
            } catch (...) {
                // Catch any exceptions from accessing modules
            }
            
            // Test extending the list
            try {
                torch::nn::ModuleList extension;
                extension->push_back(torch::nn::Linear(1, 2));
                extension->push_back(torch::nn::ReLU());
                
                module_list->extend(*extension);
            } catch (...) {
                // Catch any exceptions from extending
            }
        }
        
        // Test ModuleDict container
        {
            torch::nn::ModuleDict module_dict;
            
            // Add modules to the dictionary
            try {
                module_dict->update("linear1", torch::nn::Linear(10, 5));
                module_dict->update("relu", torch::nn::ReLU());
                module_dict->update("linear2", torch::nn::Linear(5, 1));
            } catch (...) {
                // Catch any exceptions from inserting modules
            }
            
            // Test accessing modules
            try {
                if (module_dict->contains("linear1")) {
                    auto module = module_dict["linear1"];
                }
                
                auto keys = module_dict->keys();
                auto values = module_dict->values();
            } catch (...) {
                // Catch any exceptions from accessing modules
            }
            
            // Test clear
            try {
                module_dict->clear();
            } catch (...) {
                // Catch any exceptions from clearing
            }
        }
        
        // Test ParameterList container
        {
            torch::nn::ParameterList param_list;
            
            // Add parameters to the list
            try {
                param_list->append(tensor1);
                param_list->append(tensor2);
            } catch (...) {
                // Catch any exceptions from adding parameters
            }
            
            // Test accessing parameters
            try {
                if (param_list->size() > 0) {
                    auto param = param_list[0];
                    auto params = param_list->parameters();
                }
            } catch (...) {
                // Catch any exceptions from accessing parameters
            }
            
            // Test extending the list
            try {
                torch::nn::ParameterList extension;
                extension->append(tensor3);
                
                param_list->extend(*extension);
            } catch (...) {
                // Catch any exceptions from extending
            }
        }
        
        // Test ParameterDict container
        {
            torch::nn::ParameterDict param_dict;
            
            // Add parameters to the dictionary
            try {
                param_dict->insert("param1", tensor1);
                param_dict->insert("param2", tensor2);
            } catch (...) {
                // Catch any exceptions from inserting parameters
            }
            
            // Test accessing parameters
            try {
                if (param_dict->contains("param1")) {
                    auto param = param_dict["param1"];
                }
                
                auto keys = param_dict->keys();
                auto values = param_dict->values();
            } catch (...) {
                // Catch any exceptions from accessing parameters
            }
            
            // Test clear
            try {
                param_dict->clear();
            } catch (...) {
                // Catch any exceptions from clearing
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