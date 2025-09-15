Help me reproduce the bugs in tensorflow. You should output python files and bug reports to a directory crash that will reveil a crash.

## Requirements:
- Use TensorFlow 2.19, CPU only

## Structure 
- _fuzz_result/tf2.19-fuzz-600s: the fuzzing result directory
- testharness: the test harness directory
- crash: the output directory for reproducible bugs, have exisiting bugs and place virtual env for tensorflow

## What you need to know
- The crashes is under /home/fqin2/FlashFuzz/_fuzz_result/tf2.19-fuzz-600s
- Do not reproduce the crash that you have reproduced before
- There are some common false positives, such as:
  - UndefinedBehaviorSanitizer: SEGV on unknown address 0x0000000000a0
  - FPE error in the main.cpp / fuzz.cpp
  - everything else can be a valid bug, please try to reproduce it, including dead signal, and "F ..." from tensorflow fatallog. They shouldn't crash during execution.
- Some bugs in the api has been reported, at https://github.com/tensorflow/tensorflow/issues?q=is%3Aissue%20state%3Aopen%20author%3ASilentTester73, skip these bugs
- Under the _fuzz_result directory, there are api names as the dir name and crash inputs under the artifacts directory

## Suggest to do 
- check if the bug and api has been discovered before by SilentTester73.
- create a docker `ncsuswat/flashfuzz:tf2.19-fuzz` and most thing in in the docker under /root/tensorflow/fuzz directory
- copy the crash inputs to the target api directory in the docker
- modify the fuzz.cpp / main.cpp so that you can print the input and arguments that cause the crash
- Try to write a python script to call the target api with the input and arguments that cause the crash
- Verify and it verifird, put the python script to the crash directory

## Expect output
- A directory named crash, with python files that can reproduce the bugs, api name in the file name
- Minimize the reproducible python files, for example, do not load inputs by artifact file, but construct the input in the python file
- A same {api}_bug_report.md that can submmit to github as a bug report
- It's ok if some bugs cannot be reproduced
