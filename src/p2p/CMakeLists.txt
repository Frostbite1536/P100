add_library(p100_p2p STATIC p2p.cpp)
target_include_directories(p100_p2p PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(p100_p2p PUBLIC ${P100_CUDART_TARGET})
