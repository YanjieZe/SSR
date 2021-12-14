cmake_minimum_required(VERSION 3.10)
project(linearfold-python CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(pybind11)
pybind11_add_module(pylinearfold
    interface.cpp
    # src/LinearFoldEval.cpp src/LinearFold.h
    # src/Utils/energy_parameter.h src/Utils/feature_weight.h
    # src/Utils/intl11.h src/Utils/intl21.h src/Utils/intl22.h
    # src/Utils/utility_v.h src/Utils/utility.h
    )
target_compile_features(pylinearfold PUBLIC cxx_std_17)
#target_link_libraries(mxfold2)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_LIBCPP_DEBUG=0")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXXFLAGS} -D_GLIBCXX_DEBUG")
