#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <cstdlib>        // For getenv, setenv, unsetenv
#include <unistd.h>       // For getcwd, chdir
#include <climits>        // For PATH_MAX
#include <sys/utsname.h>  // For uname

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Test executable path (using /proc/self/exe on Linux)
        char exe_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len != -1) {
            exe_path[len] = '\0';
            std::string executable_path(exe_path);
        }
        
        // Test environment variable operations
        if (offset < Size) {
            uint8_t env_var_selector = Data[offset++];
            std::vector<std::string> env_vars = {"PATH", "HOME", "USER", "PYTHONPATH", "LD_LIBRARY_PATH"};
            std::string env_var = env_vars[env_var_selector % env_vars.size()];
            const char* env_value = getenv(env_var.c_str());
            if (env_value) {
                std::string env_str(env_value);
            }
        }
        
        // Test set_env and get_env
        if (offset + 1 < Size) {
            uint8_t test_var_selector = Data[offset++];
            std::vector<std::string> test_vars = {"TEST_VAR1", "TEST_VAR2", "FUZZ_TEST_VAR"};
            std::string test_var = test_vars[test_var_selector % test_vars.size()];
            
            // Create a value from the remaining data
            std::string test_value;
            size_t value_len = std::min(Size - offset, static_cast<size_t>(10));
            for (size_t i = 0; i < value_len; i++) {
                test_value += static_cast<char>(Data[offset + i] % 95 + 32); // Printable ASCII
            }
            offset += value_len;
            
            // Set and get the environment variable
            setenv(test_var.c_str(), test_value.c_str(), 1);
            const char* retrieved_value = getenv(test_var.c_str());
            if (retrieved_value) {
                std::string retrieved_str(retrieved_value);
            }
        }
        
        // Test unset_env
        if (offset < Size) {
            uint8_t unset_var_selector = Data[offset++];
            std::vector<std::string> unset_vars = {"FUZZ_UNSET_VAR1", "FUZZ_UNSET_VAR2"};
            std::string unset_var = unset_vars[unset_var_selector % unset_vars.size()];
            
            // First set it
            setenv(unset_var.c_str(), "value_to_unset", 1);
            
            // Then unset it
            unsetenv(unset_var.c_str());
            
            // Verify it's unset
            const char* unset_value = getenv(unset_var.c_str());
        }
        
        // Test get_cwd and set_cwd
        char cwd_buffer[PATH_MAX];
        char* original_cwd = getcwd(cwd_buffer, sizeof(cwd_buffer));
        if (original_cwd) {
            std::string original_cwd_str(original_cwd);
            
            // Try to set cwd to a different directory if we have enough data
            if (offset < Size) {
                uint8_t dir_selector = Data[offset++];
                std::vector<std::string> test_dirs = {".", "..", "/tmp", "/home"};
                std::string test_dir = test_dirs[dir_selector % test_dirs.size()];
                
                try {
                    if (chdir(test_dir.c_str()) == 0) {
                        char new_cwd_buffer[PATH_MAX];
                        char* new_cwd = getcwd(new_cwd_buffer, sizeof(new_cwd_buffer));
                        if (new_cwd) {
                            std::string new_cwd_str(new_cwd);
                        }
                        
                        // Restore original working directory
                        chdir(original_cwd_str.c_str());
                    }
                } catch (const std::exception& e) {
                    // Directory might not exist or be accessible, that's fine
                }
            }
        }
        
        // Test get_os_name
        struct utsname sys_info;
        if (uname(&sys_info) == 0) {
            std::string os_name(sys_info.sysname);
        }
        
        // Test get_platform_name
        if (uname(&sys_info) == 0) {
            std::string platform_name(sys_info.machine);
        }
        
        // Test Python-related functions (using environment variables as fallback)
        const char* python_version = getenv("PYTHON_VERSION");
        if (python_version) {
            std::string python_version_str(python_version);
        }
        
        const char* python_path = getenv("PYTHONPATH");
        if (python_path) {
            std::string python_path_str(python_path);
        }
        
        const char* python_executable = getenv("PYTHON_EXECUTABLE");
        if (python_executable) {
            std::string python_executable_str(python_executable);
        }
        
        const char* python_prefix = getenv("PYTHON_PREFIX");
        if (python_prefix) {
            std::string python_prefix_str(python_prefix);
        }
        
        const char* python_include_dir = getenv("PYTHON_INCLUDE_DIR");
        if (python_include_dir) {
            std::string python_include_dir_str(python_include_dir);
        }
        
        const char* python_lib_dir = getenv("PYTHON_LIB_DIR");
        if (python_lib_dir) {
            std::string python_lib_dir_str(python_lib_dir);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
