#include "gl_utils.h"
#include <fstream>
#include <sstream>
#include <cstdio>

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Failed to open: %s\n", path.c_str());
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!vs || !fs) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    int ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

GLuint loadProgramFromFiles(const std::string& vertPath, const std::string& fragPath) {
    std::string vs = readFile(vertPath);
    std::string fs = readFile(fragPath);
    return createProgram(vs.c_str(), fs.c_str());
}
