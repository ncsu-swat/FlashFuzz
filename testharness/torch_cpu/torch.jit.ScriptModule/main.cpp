#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
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
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get some fuzzer-controlled values for module behavior
        uint8_t module_type = (Size > offset) ? Data[offset % Size] % 4 : 0;
        (void)module_type; // Suppress unused variable warning
        
        // Create a ScriptModule
        torch::jit::Module module("fuzz_module");
        
        // Register a parameter based on input tensor shape
        try {
            torch::Tensor param = torch::randn_like(input_tensor);
            module.register_parameter("weight", param, false);
        } catch (...) {
            // Ignore parameter registration failures
        }
        
        // Register a buffer
        try {
            torch::Tensor buffer = torch::zeros_like(input_tensor);
            module.register_buffer("running_mean", buffer);
        } catch (...) {
            // Ignore buffer registration failures
        }
        
        // Test various Module methods
        
        // Test attribute access
        try {
            bool has_param = module.hasattr("weight");
            if (has_param) {
                torch::jit::IValue attr_val = module.attr("weight");
                if (attr_val.isTensor()) {
                    torch::Tensor weight = attr_val.toTensor();
                    (void)weight;
                }
            }
        } catch (...) {
            // Expected for missing attributes
        }
        
        // Test named_parameters
        try {
            for (const auto& param : module.named_parameters()) {
                std::string name = param.name;
                torch::Tensor value = param.value;
                (void)name;
                (void)value;
            }
        } catch (...) {
            // Ignore iteration errors
        }
        
        // Test named_buffers
        try {
            for (const auto& buffer : module.named_buffers()) {
                std::string name = buffer.name;
                torch::Tensor value = buffer.value;
                (void)name;
                (void)value;
            }
        } catch (...) {
            // Ignore iteration errors
        }
        
        // Test named_attributes
        try {
            for (const auto& attr : module.named_attributes()) {
                std::string name = attr.name;
                (void)name;
            }
        } catch (...) {
            // Ignore iteration errors
        }
        
        // Test clone
        try {
            torch::jit::Module cloned = module.clone();
            bool cloned_has_param = cloned.hasattr("weight");
            (void)cloned_has_param;
        } catch (...) {
            // Ignore clone errors
        }
        
        // Test copy_ method
        try {
            torch::jit::Module target("target_module");
            target.register_parameter("weight", torch::randn_like(input_tensor), false);
            // copy_ not always available, try if it exists
        } catch (...) {
            // Ignore copy errors
        }
        
        // Test to() for device conversion (modifies in-place, returns void)
        try {
            module.to(torch::kCPU);
        } catch (...) {
            // Ignore device transfer errors
        }
        
        // Test to() for dtype conversion (modifies in-place, returns void)
        try {
            module.to(torch::kFloat32);
        } catch (...) {
            // Ignore dtype conversion errors
        }
        
        // Test eval() and train() modes
        try {
            module.eval();
            bool is_training_after_eval = module.is_training();
            (void)is_training_after_eval;
            
            module.train(true);
            bool is_training_after_train = module.is_training();
            (void)is_training_after_train;
        } catch (...) {
            // Ignore mode setting errors
        }
        
        // Test named_modules
        try {
            for (const auto& submod : module.named_modules()) {
                std::string name = submod.name;
                (void)name;
            }
        } catch (...) {
            // Ignore iteration errors
        }
        
        // Test named_children
        try {
            for (const auto& child : module.named_children()) {
                std::string name = child.name;
                (void)name;
            }
        } catch (...) {
            // Ignore iteration errors
        }
        
        // Test registering a submodule
        try {
            torch::jit::Module submodule("submodule");
            submodule.register_parameter("sub_weight", torch::randn({2, 2}), false);
            module.register_module("child", submodule);
            
            // Access the child
            if (module.hasattr("child")) {
                torch::jit::IValue child_val = module.attr("child");
                (void)child_val;
            }
        } catch (...) {
            // Ignore submodule errors
        }
        
        // Test get_methods if any methods exist
        try {
            auto method_names = module.get_methods();
            for (const auto& method : method_names) {
                std::string name = method.name();
                (void)name;
            }
        } catch (...) {
            // Ignore method access errors
        }
        
        // Test parameters() iterator
        try {
            auto params = module.parameters();
            for (const auto& p : params) {
                torch::Tensor param_tensor = p;
                (void)param_tensor;
            }
        } catch (...) {
            // Ignore errors
        }
        
        // Test buffers() iterator
        try {
            auto bufs = module.buffers();
            for (const auto& b : bufs) {
                torch::Tensor buf_tensor = b;
                (void)buf_tensor;
            }
        } catch (...) {
            // Ignore errors
        }
        
        // Test dump() for debugging info
        try {
            std::string dump_str = module.dump_to_str(true, false, false);
            (void)dump_str;
        } catch (...) {
            // Ignore dump errors
        }
        
        // Test deepcopy
        try {
            torch::jit::Module deep_copy = module.deepcopy();
            (void)deep_copy;
        } catch (...) {
            // Ignore deepcopy errors
        }
        
        // Create second tensor if we have more data
        if (Size > offset + 4) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test module with multiple parameters
            try {
                torch::jit::Module multi_param_module("multi_param");
                multi_param_module.register_parameter("weight1", input_tensor.clone(), false);
                multi_param_module.register_parameter("weight2", second_tensor.clone(), false);
                multi_param_module.register_buffer("buffer1", torch::zeros_like(input_tensor));
                
                // Iterate all parameters
                int param_count = 0;
                for (const auto& p : multi_param_module.named_parameters()) {
                    param_count++;
                    (void)p;
                }
                (void)param_count;
            } catch (...) {
                // Ignore errors
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