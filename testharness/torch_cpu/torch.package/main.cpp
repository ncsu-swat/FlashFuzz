#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/script.h>
#include <filesystem>
#include <fstream>

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
        
        // Create a tensor to package
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple module to package
        torch::jit::Module module("test_module");
        
        // Register the tensor as a buffer in the module
        module.register_buffer("test_tensor", tensor);
        
        // Create a temporary directory for package
        std::string temp_dir = "/tmp/torch_package_fuzz";
        std::filesystem::create_directories(temp_dir);
        std::string package_path = temp_dir + "/test_package.pt";
        
        // Try different packaging modes based on input data
        if (offset < Size) {
            uint8_t mode_selector = Data[offset++];
            
            // Select different save approaches
            if (mode_selector % 2 == 0) {
                // Save the module to a package
                module.save(package_path);
            } else {
                // Use the torch::jit::save function
                torch::jit::save(module, package_path);
            }
        } else {
            // Default case
            module.save(package_path);
        }
        
        // Try to load the package back
        if (std::filesystem::exists(package_path)) {
            try {
                torch::jit::Module loaded_module = torch::jit::load(package_path);
                
                // Access the tensor from the loaded module
                torch::Tensor loaded_tensor = loaded_module.attr("test_tensor").toTensor();
                
                // Verify the loaded tensor matches the original
                if (tensor.sizes() != loaded_tensor.sizes() || 
                    tensor.dtype() != loaded_tensor.dtype()) {
                    throw std::runtime_error("Loaded tensor doesn't match original");
                }
            } catch (const c10::Error& e) {
                // Expected errors when loading invalid packages
            }
            
            // Clean up
            std::filesystem::remove(package_path);
        }
        
        // Try more complex packaging if we have more data
        if (offset + 4 < Size) {
            // Create another tensor
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a more complex module with a method
            torch::jit::Module module2("test_module2");
            module2.register_buffer("test_tensor", tensor);
            module2.register_buffer("test_tensor2", tensor2);
            
            // Create a package with different options
            std::string package_path2 = temp_dir + "/test_package2.pt";
            
            // Save the module
            torch::jit::save(module2, package_path2);
            
            // Try to load it back
            if (std::filesystem::exists(package_path2)) {
                try {
                    torch::jit::Module loaded_module = torch::jit::load(package_path2);
                } catch (const c10::Error& e) {
                    // Expected errors when loading invalid packages
                }
                
                // Clean up
                std::filesystem::remove(package_path2);
            }
        }
        
        // Clean up the temporary directory
        std::filesystem::remove_all(temp_dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
