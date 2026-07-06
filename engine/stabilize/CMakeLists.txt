# Drop this directory into llama.cpp's examples/ as `redraft-stabilize/` and add
# `add_subdirectory(redraft-stabilize)` to examples/CMakeLists.txt, then build the
# target `llama-redraft-stabilize`. stabilize.hpp and batched_stabilize.hpp must
# sit alongside stabilize_llama.cpp (they are the reference core, kept identical
# to the copies under engine/stabilize/ in this repo). See engine/BUILD_NOTES.md.
set(TARGET llama-redraft-stabilize)
add_executable(${TARGET} stabilize_llama.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
