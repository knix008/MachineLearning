cmake_minimum_required(VERSION 3.16)
project(QtCalculator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Qt6 components
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# Set up Qt6
qt6_standard_project_setup()

# Add executable
qt6_add_executable(calculator
    main.cpp
    calculator.cpp
    calculator.h
)

# Link Qt6 libraries
target_link_libraries(calculator Qt6::Core Qt6::Widgets)

# Enable Qt6 MOC (Meta-Object Compiler)
set_target_properties(calculator PROPERTIES
    AUTOMOC ON
)
