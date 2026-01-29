#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// Define a simple C++ class to register with torch.classes
class MyStackClass : public torch::CustomClassHolder {
private:
    std::vector<torch::Tensor> stack_;

public:
    MyStackClass() {}

    void push(torch::Tensor x) {
        stack_.push_back(x);
    }

    torch::Tensor pop() {
        if (stack_.empty()) {
            throw std::runtime_error("Empty stack");
        }
        auto ret = stack_.back();
        stack_.pop_back();
        return ret;
    }

    std::vector<torch::Tensor> getStack() {
        return stack_;
    }

    int64_t size() {
        return stack_.size();
    }

    void clear() {
        stack_.clear();
    }
};

// Register the custom C++ class with torch.classes at static initialization time
static auto registered = torch::class_<MyStackClass>("_TorchScriptTesting", "MyStackClass")
    .def(torch::init<>())
    .def("push", &MyStackClass::push)
    .def("pop", &MyStackClass::pop)
    .def("getStack", &MyStackClass::getStack)
    .def("size", &MyStackClass::size)
    .def("clear", &MyStackClass::clear);

// --- Fuzzer Entry Point ---
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
        
        // Ensure we have enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create an instance of our custom class
        auto stack_instance = c10::make_intrusive<MyStackClass>();
        
        // Determine how many tensors to create and push to the stack
        uint8_t num_tensors = Data[offset++] % 5 + 1; // 1-5 tensors
        
        // Create and push tensors to the stack
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                stack_instance->push(tensor);
            } catch (const std::exception&) {
                // Continue with the next tensor if one fails
                continue;
            }
        }
        
        // Test various operations on the stack
        if (stack_instance->size() > 0) {
            // Pop a tensor if the stack is not empty
            auto popped = stack_instance->pop();
            
            // Get the current stack
            auto stack = stack_instance->getStack();
            
            // Push a tensor with different properties
            if (offset + 2 < Size) {
                try {
                    auto another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    stack_instance->push(another_tensor);
                } catch (const std::exception&) {
                    // Ignore tensor creation failures
                }
            }
            
            // Test edge case: pop until empty
            while (stack_instance->size() > 0) {
                stack_instance->pop();
            }
            
            // Test expected exception on empty stack pop
            if (Data[0] % 2 == 0) {
                try {
                    stack_instance->pop(); // Should throw
                } catch (const std::runtime_error&) {
                    // Expected exception, continue
                }
            }
        }
        
        // Create a new instance and test with potentially different tensor configurations
        auto another_instance = c10::make_intrusive<MyStackClass>();
        
        if (offset < Size) {
            try {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                another_instance->push(tensor);
                
                // Verify size tracking
                if (another_instance->size() != 1) {
                    return 0; // Sanity check
                }
                
                // Get stack and verify
                auto stack = another_instance->getStack();
                
                // Test clear operation
                another_instance->clear();
                
                if (another_instance->size() != 0) {
                    return 0; // Sanity check
                }
            } catch (const std::exception&) {
                // Continue execution for inner exceptions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}