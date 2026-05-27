from OpenGL.GL import glDrawElements
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

def check_gl_errors():
    """Check for OpenGL errors and print them if any exist."""
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL Error: {gluErrorString(error).decode('utf-8')}")


class renderable:
     def __init__(self, vao, n_verts, n_faces,texture_id):
         self.vao = vao
         self.n_verts = n_verts
         self.n_faces = n_faces
         self.texture_id = texture_id
         
     vao = None #vertex array object
     n_verts = None
     n_faces = None

class shader:
    def __init__(self, vertex_shader_str , fragment_shader_str, geometry_shader_str=None):
        self.uniforms = {}
        shaders = [compileShader(vertex_shader_str, GL_VERTEX_SHADER)]
        if geometry_shader_str is not None:
            shaders.append(compileShader(geometry_shader_str, GL_GEOMETRY_SHADER))
        shaders.append(compileShader(fragment_shader_str, GL_FRAGMENT_SHADER))
        self.program = compileProgram(*shaders)
    def uni(self, name):
            if name not in self.uniforms:
                self.uniforms[name] = glGetUniformLocation(self.program, name)
            return self.uniforms[name]
    

