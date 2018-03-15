# Facedetector based on dlib
# Copyright (C) 2018 Ingo Ruhnke <grumbel@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.0)
project(facedetector VERSION 0.1.0)

option(WARNINGS "Switch on extra warnings" OFF)
option(WERROR "Treat all warnings as errors" OFF)


if(WARNINGS)
  set(WARNINGS_CXX_FLAGS ${WARNINGS_CXX_FLAGS}
    -pedantic -Wall -Wextra -Wno-c++0x-compat -Wnon-virtual-dtor -Weffc++
    -Wshadow -Wcast-qual -Winit-self -Wno-unused-parameter -Wunreachable-code)
  # clang: -Wconversion
  # -Winline -Wfloat-equal
endif()

if(WERROR)
  set(WARNINGS_CXX_FLAGS ${WARNINGS_CXX_FLAGS}
    -Werror)
endif()

add_compile_options(-std=c++14)

find_package(Threads REQUIRED)

add_executable(facedetector facedetector.cpp)
target_compile_options(facedetector PUBLIC ${WARNINGS_CXX_FLAGS})
target_link_libraries(facedetector lapack blas dlib fmt ${CMAKE_THREAD_LIBS_INIT})

# EOF #
