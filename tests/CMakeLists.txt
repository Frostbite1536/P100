foreach(name test_fp16 test_reference test_tiled_model error_budget test_tiled_model_sm60 error_budget_sm60)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE p100_ref)
endforeach()
add_test(NAME fp16 COMMAND test_fp16)
foreach(case tiny uniform double gqa causal masked softcap large validation)
  add_test(NAME reference_${case} COMMAND test_reference ${case})
endforeach()
foreach(case matrix masked rounding validation)
  add_test(NAME tiled_${case} COMMAND test_tiled_model ${case})
endforeach()
add_test(NAME error_budget COMMAND error_budget)
add_test(NAME test_tiled_model_sm60 COMMAND test_tiled_model_sm60)
# The full Release run exceeds 120 s; --long includes the 32768-key shapes.
add_test(NAME error_budget_sm60 COMMAND error_budget_sm60)


# Needs the CUDA runtime (top-level block); skipped on CPU-only builds.
if(TARGET p100_p2p)
  add_executable(test_p2p test_p2p.cpp)
  target_link_libraries(test_p2p PRIVATE p100_p2p)
  foreach(case probe env staged_sizes stream_order_before buffer_reuse peer)
    add_test(NAME p2p_${case} COMMAND test_p2p ${case})
    set_tests_properties(p2p_${case} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()

# Static sm_60 controls use the active .venv (including the main checkout's
# interpreter in worktrees), with the same wheel discovery as the report tool.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import ctypes, runpy, sys; tool = runpy.run_path(sys.argv[1]); ctypes.CDLL(str(tool['NV'] / 'cuda_nvrtc/lib/libnvrtc.so.12')); sys.exit(0 if tool['PTXAS'].is_file() else 1)"
      ${PROJECT_SOURCE_DIR}/bin/sm60-report.py
    RESULT_VARIABLE _sm60_toolchain_result
    OUTPUT_QUIET ERROR_QUIET)
  if(_sm60_toolchain_result EQUAL 0)
    foreach(control ok spill smem)
      add_test(NAME sm60_report_${control}
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bin/sm60-report.py
                ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/sm60/${control}.cu)
    endforeach()
    set_tests_properties(sm60_report_spill sm60_report_smem PROPERTIES WILL_FAIL TRUE)
    foreach(control dp4a wmma)
      # Explicit paths bypass the default fixture exclusion in the checker.
      configure_file(fixtures/sm60/${control}.cu sm60/${control}.cu COPYONLY)
      add_test(NAME forbidden_${control}
        COMMAND bash ${PROJECT_SOURCE_DIR}/bin/check-forbidden.sh
                ${CMAKE_CURRENT_BINARY_DIR}/sm60/${control}.cu)
      set_tests_properties(forbidden_${control} PROPERTIES WILL_FAIL TRUE)
    endforeach()
  else()
    message(STATUS "P100: .venv NVRTC/ptxas unavailable; skipping sm_60 tooling controls")
  endif()
else()
  message(STATUS "P100: Python unavailable; skipping sm_60 tooling controls")
endif()

# bench/analyze.py must parse real test-backend-ops SQL output (fixture: CPU run at bench/LLAMA_COMMIT).
if(Python3_Interpreter_FOUND)
  add_test(NAME analyze_tbo
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bench/analyze.py tbo
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tbo-perf-cpu.sql ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tbo-perf-cpu.sql)
  set_tests_properties(analyze_tbo PROPERTIES PASS_REGULAR_EXPRESSION "type_V=q8_0 \\| [0-9.]+ ms \\| [0-9.]+ ms \\| 1\\.00×")
  # llama-bench rounds (bench/patches/run.sh): a comma-joined group is summarised as median (min-max, n).
  add_test(NAME analyze_lb_rounds
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bench/analyze.py lb
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r1.jsonl,${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r2.jsonl
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r3.jsonl --labels a,b)
  set_tests_properties(analyze_lb_rounds PROPERTIES PASS_REGULAR_EXPRESSION "105\\.00 \\(100\\.00–110\\.00, n=2\\) \\| 130\\.00 ± 1\\.00 \\| \\+23\\.8%")
endif()

# Fixture commands validate the Phase 0 and patch harness plumbing without CUDA,
# llama.cpp, models, worktree writes, or persistent benchmark output.
if(Python3_Interpreter_FOUND)
  find_program(P100_BASH_EXECUTABLE bash REQUIRED)
  foreach(kind phase0 patches)
    add_test(NAME ${kind}_dryrun
      COMMAND ${CMAKE_COMMAND} -E env "PYTHON=${Python3_EXECUTABLE}"
              ${P100_BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/shims/dryrun
              ${PROJECT_SOURCE_DIR} ${kind})
    set_tests_properties(${kind}_dryrun PROPERTIES TIMEOUT 60)
  endforeach()
endif()

# Decode predictions and compile-only llama.cpp dot-product SASS fixtures.
if(Python3_Interpreter_FOUND)
  add_test(NAME decode_model
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/decode_model_check.py)
  if(DEFINED ENV{P100_NVCC})
    set(_vecdot_nvcc "$ENV{P100_NVCC}")
  else()
    set(_vecdot_nvcc "/home/p51/Github/P100/tools/cuda-12.8/bin/nvcc")
  endif()
  if(EXISTS "${_vecdot_nvcc}")
    add_test(NAME vecdot_sass
      COMMAND ${CMAKE_COMMAND} -E env "P100_NVCC=${_vecdot_nvcc}" "PYTHON=${Python3_EXECUTABLE}"
              ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/decode_model_check.py --sass)
    set_tests_properties(vecdot_sass PROPERTIES TIMEOUT 180)
  endif()
endif()
