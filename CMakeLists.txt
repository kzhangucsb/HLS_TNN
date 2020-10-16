cmake_minimum_required(VERSION 3.0.0)
project(tnn_mnist VERSION 0.1.0)

include(CTest)
enable_testing()
file(GLOB Sources "*.cpp")

add_executable(tnn_mnist ${Sources})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
