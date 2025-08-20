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
        
        // Create a tensor to work with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get some bytes to determine library name and function name
        if (offset + 2 >= Size) {
            return 0;
        }
        
        uint8_t lib_name_selector = Data[offset++];
        uint8_t func_name_selector = Data[offset++];
        
        // Create a simple library name
        std::string library_name = "fuzz_lib_" + std::to_string(lib_name_selector % 10);
        
        // Create a function name
        std::string function_name = "fuzz_func_" + std::to_string(func_name_selector % 10);
        
        // Test torch::library functionality
        // Define a custom library using TORCH_LIBRARY macro approach
        static bool library_registered = false;
        if (!library_registered) {
            auto m = TORCH_LIBRARY(fuzz_lib, m) {
                m.def("test_op(Tensor x) -> Tensor", [](const torch::Tensor& x) -> torch::Tensor {
                    return x.clone();
                });
            };
            library_registered = true;
        }
        
        // Test calling a registered function using torch::jit::get_operation
        try {
            auto op = torch::jit::get_operation("fuzz_lib::test_op");
            if (op) {
                torch::jit::Stack stack;
                stack.push_back(tensor);
                op->getOperation()(stack);
                if (!stack.empty()) {
                    torch::Tensor result = stack.back().toTensor();
                }
            }
        } catch (const std::exception&) {
            // Operation might not be found
        }
        
        // Test registering a function with different dispatch key
        if (offset < Size) {
            uint8_t dispatch_key_selector = Data[offset++];
            
            // Test with CPU implementation
            static bool cpu_impl_registered = false;
            if (!cpu_impl_registered && dispatch_key_selector % 2 == 0) {
                auto m = TORCH_LIBRARY_IMPL(fuzz_lib, CPU, m) {
                    m.impl("test_op", [](const torch::Tensor& x) -> torch::Tensor {
                        return x * 2;
                    });
                };
                cpu_impl_registered = true;
            }
        }
        
        // Test creating another library with different operations
        if (offset < Size) {
            static bool second_library_registered = false;
            if (!second_library_registered) {
                auto m = TORCH_LIBRARY(fuzz_lib2, m) {
                    m.def("overloaded(Tensor x) -> Tensor");
                    m.def("overloaded(Tensor x, Tensor y) -> Tensor");
                };
                
                auto impl = TORCH_LIBRARY_IMPL(fuzz_lib2, CPU, m) {
                    m.impl("overloaded", [](const torch::Tensor& x) -> torch::Tensor {
                        return x.clone();
                    });
                    m.impl("overloaded", [](const torch::Tensor& x, const torch::Tensor& y) -> torch::Tensor {
                        return x + y;
                    });
                };
                second_library_registered = true;
            }
            
            // Try to call the overloaded functions
            try {
                auto op1 = torch::jit::get_operation("fuzz_lib2::overloaded");
                if (op1) {
                    torch::jit::Stack stack1;
                    stack1.push_back(tensor);
                    op1->getOperation()(stack1);
                }
                
                // Try with two tensors if we have enough data
                if (offset < Size) {
                    torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::jit::Stack stack2;
                    stack2.push_back(tensor);
                    stack2.push_back(tensor2);
                    if (op1) {
                        op1->getOperation()(stack2);
                    }
                }
            } catch (const std::exception&) {
                // Operations might not be found or might fail
            }
        }
        
        // Test creating a library with different kinds using manual registration
        if (offset < Size) {
            uint8_t kind_selector = Data[offset++] % 3;
            
            static bool kind_library_registered = false;
            if (!kind_library_registered) {
                switch (kind_selector) {
                    case 0: {
                        auto m = TORCH_LIBRARY(fuzz_kind_def, m) {
                            m.def("kind_op(Tensor x) -> Tensor");
                        };
                        break;
                    }
                    case 1: {
                        auto m = TORCH_LIBRARY_IMPL(fuzz_kind_impl, CPU, m) {
                            // Implementation only
                        };
                        break;
                    }
                    case 2: {
                        auto m = TORCH_LIBRARY_FRAGMENT(fuzz_kind_fragment, m) {
                            m.def("fragment_op(Tensor x) -> Tensor", [](const torch::Tensor& x) -> torch::Tensor {
                                return x.clone();
                            });
                        };
                        break;
                    }
                }
                kind_library_registered = true;
            }
        }
        
        // Test some basic tensor operations to ensure everything works
        torch::Tensor test_result = tensor.clone();
        if (tensor.numel() > 0) {
            test_result = tensor + 1;
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}