#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple module
        torch::jit::Module module("test_module");
        
        // Create a tensor to add as an attribute
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a name for the attribute
        std::string attr_name = "attr";
        if (offset < Size) {
            uint8_t name_len = Data[offset++] % 10 + 1; // 1-10 characters
            if (offset + name_len <= Size) {
                attr_name = std::string(reinterpret_cast<const char*>(Data + offset), name_len);
                offset += name_len;
            }
        }
        
        // Register the tensor as a parameter
        module.register_parameter(attr_name, tensor, false);
        
        // Create a submodule
        torch::jit::Module submodule("submodule");
        
        // Create another tensor for the submodule
        if (offset < Size) {
            torch::Tensor sub_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            submodule.register_parameter("sub_attr", sub_tensor, false);
        }
        
        // Add the submodule to the main module
        module.register_module("sub", submodule);
        
        // Test module replacement functionality
        if (offset < Size) {
            // Create a new module to replace the submodule
            torch::jit::Module new_module("new_module");
            
            // Add a parameter to the new module
            torch::Tensor new_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            new_module.register_parameter("new_attr", new_tensor, false);
            
            // Use _set_module to replace the submodule (internal method)
            module._set_module("sub", new_module);
            
            // Verify the replacement worked by accessing the module
            auto modules = module.named_modules();
            for (const auto& named_module : modules) {
                if (named_module.name == "sub") {
                    auto params = named_module.value.named_parameters();
                    break;
                }
            }
        }
        
        // Try to set a non-existent module
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::jit::Module empty_module("empty");
            try {
                module._set_module("nonexistent", empty_module);
            } catch (const c10::Error&) {
                // Expected exception for non-existent module
            }
        }
        
        // Try to set a module with the same name as a parameter
        if (offset < Size && Data[offset] % 2 == 1) {
            torch::jit::Module conflict_module("conflict");
            try {
                module._set_module(attr_name, conflict_module);
            } catch (const c10::Error&) {
                // Expected exception for name conflict
            }
        }
        
        // Test serialization and deserialization
        std::stringstream ss;
        module.save(ss);
        ss.seekg(0);
        torch::jit::Module loaded_module = torch::jit::load(ss);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}