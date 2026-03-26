#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <glad/gl.h>
#include <string>

GLuint compileShader(GLenum type, const char* source);
GLuint createProgram(const char* vertSrc, const char* fragSrc);
GLuint loadProgramFromFiles(const std::string& vertPath, const std::string& fragPath);
std::string readFile(const std::string& path);
