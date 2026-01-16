
import pygame
import json
import pymeshlab
from pygame.locals import *

from OpenGL.GL import glDrawElements
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image
import ctypes

import glm

import imgui
from imgui.integrations.pygame import PygameRenderer
from tkinter import Tk, filedialog
from  renderable import * 
import numpy as np
import os
import ctypes
import numpy as np
import trackball 
import texture
import metashape_loader
import fbo
import   shaders 


import xml.etree.ElementTree as ET
import numpy as np
import zipfile

from plane import fit_plane, project_point_on_plane
from ctypes import c_uint32, cast, POINTER
import zip_utils

import sys 

from  detector import apply_yolo

import pandas as pd
import os
from collections import Counter


curr_camera_id = 0

def create_buffers_frame():
    
    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aPosition')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    verts = [0,0,0, 1,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,1]
    verts = np.array(verts, dtype=np.float32)

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,verts.nbytes, verts, GL_STATIC_DRAW)
    
    # Generate buffers to hold our vertices
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aColor')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    col = [1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,0,1, 0,0,1]
    col = np.array(col, dtype=np.float32)

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,col.nbytes, col, GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    glBindVertexArray( 0 )
    
    # Unbind other stuff
    glDisableVertexAttribArray(position)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vertex_array_object
     
def create_buffers_fsq():
        
    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader_fsq.program, 'aPosition')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    verts = [-1,-1,0, 1,-1,0, 1,1,0, -1,-1,0, 1,1,0, -1,1,0] 
    verts = np.array(verts, dtype=np.float32)
 
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,verts.nbytes, verts, GL_STATIC_DRAW)
    
    # Unbind other stuff
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vertex_array_object

def create_buffers(verts,wed_tcoord,inds):
    vert_pos            = np.zeros((len(inds) * 3,  3), dtype=np.float32)
    tcoords             = np.zeros((len(inds) * 3,  2), dtype=np.float32)
    for i in range(len(inds)):
        vert_pos[i*3] = verts[inds[i,0]]
        vert_pos[i*3+1] = verts[inds[i,1]]
        vert_pos[i*3+2] = verts[inds[i,2]]

        tcoords [i * 3  ] = wed_tcoord[i*3   ]
        tcoords [i * 3+1] = wed_tcoord[i*3+1 ]
        tcoords [i * 3+2] = wed_tcoord[i*3+2 ]

    vert_pos = vert_pos.flatten()
    tcoords = tcoords.flatten()

    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_object )
    
    # Generate buffers to hold our vertices
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    
    # Get the position of the 'position' in parameter of our shader and bind it.
    glEnableVertexAttribArray(shaders.aPOSITION_LOC)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(shaders.aPOSITION_LOC, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,vert_pos.nbytes, vert_pos, GL_STATIC_DRAW)
    
    # Generate buffers to hold our texcoord
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    
    # Get the position of the 'texcoord' in parameter of our shader and bind it.
    glEnableVertexAttribArray(shaders.aTEXCOORD_LOC)
    
    # Describe the texcoord data layout in the buffer
    glVertexAttribPointer(shaders.aTEXCOORD_LOC, 2, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,tcoords.nbytes, tcoords, GL_STATIC_DRAW)

    # Create an array of n*3 elements as described
    n = len(vert_pos)
    triangle_ids = np.repeat(np.arange(n), 3).astype(np.float32).reshape(-1, 3).flatten()

    # Generate buffers to hold our triangle ids
    triangle_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, triangle_buffer)

    # Get the position of the 'aIdTriangle' in parameter of our shader and bind it.
    glEnableVertexAttribArray(shaders.aIDTRIANGLE_LOC)

    # Describe the triangle id data layout in the buffer
    glVertexAttribPointer(shaders.aIDTRIANGLE_LOC, 1, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER, triangle_ids.nbytes, triangle_ids, GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    glBindVertexArray( 0 )
    
    # Unbind other stuff
    glDisableVertexAttribArray(shaders.aIDTRIANGLE_LOC)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    return vertex_array_object



#class clickable:
#    def __init__(self, id, r):
#        self.id = id
#        self.type = None
#        self.r = r




def display_image(chunk):
        global mask_zoom
        global mask_xpos
        global mask_ypos
        global W
        global H
        global curr_zoom
        global curr_center
        global curr_tra
        global tra_xstart
        global tra_ystart
        

        sensor = sensors[chunk.cameras[curr_camera_id].sensor_id]
        

        current_unit = glGetIntegerv(GL_ACTIVE_TEXTURE)
        current_texture = glGetIntegerv(GL_TEXTURE_BINDING_2D)
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glClearColor (1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_fsq.program)
        glUniform1i(shader_fsq.uni("resolution_width"), sensor.resolution["width"])
        glUniform1i(shader_fsq.uni("resolution_height"), sensor.resolution["height"])


        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        glUniform1i(shader_fsq.uni("uColorTex"),0)

        # Get the currently bound texture on GL_TEXTURE_2D

        glBindVertexArray(vao_fsq )
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0 )

       
        # Get the current zoom and center
        c = glm.vec2(mask_xpos / float(W) * 2.0 - 1.0,(H - mask_ypos) / float(H) * 2.0 - 1.0)

        if is_translating:
            curr_tra = c -  glm.vec2(tra_xstart / float(W) * 2.0 - 1.0,(H - tra_ystart) / float(H) * 2.0 - 1.0)
        

        # Apply the mask zoom and center to the current zoom and cent

        curr_zoom = curr_zoom * mask_zoom
        curr_center = (curr_center-c)*mask_zoom + c  
        tra = curr_center + curr_tra

        if not is_translating:
            curr_center = tra
            curr_tra = glm.vec2(0.0, 0.0)


        t1 = curr_zoom * 1.0 + curr_center.x + tra.x < 1.0
        t2 = curr_zoom * 1.0 + curr_center.y + tra.y < 1.0
        t3 = curr_zoom * -1.0 + curr_center.x + tra.x > -1.0
        t4 = curr_zoom * -1.0 + curr_center.y + tra.y > - 1.0

        if t1 or t2 or t3 or t4:
            curr_zoom = 1.0
            curr_center = glm.vec2(0.0, 0.0)
            curr_tra = glm.vec2(0.0, 0.0)
            tra = glm.vec2(0.0, 0.0)
           
        mask_zoom = 1.0

        # Set the zoom and center for the full screen quad shader   

        glUniform1f(shader_fsq.uni("uSca"), curr_zoom )
        glUniform2f(shader_fsq.uni("uTra"), tra.x, tra.y)  


        glActiveTexture(current_unit)
        glBindTexture(GL_TEXTURE_2D, current_texture)
        glUseProgram(0)

def chunk_matrix(chunk):
    # take care of the default values
    chunk_rot = [1,0,0,0,1,0,0,0,1]
    chunk_transl = [0,0,0]
    chunk_scal = 1

    if not chunk.rotation is None:
        chunk_rot = chunk.rotation
    if not chunk.translation is None:
        chunk_transl = chunk.translation
    if not chunk.scaling is None:
        chunk_scal = chunk.scaling

    chunk_rot = np.array(chunk_rot)
    chunk_transl = np.array(chunk_transl)

    mat4_np = np.eye(4)
    mat4_np[:3, :3] = chunk_rot.reshape(3, 3)

    chunk_rot_matrix =  glm.transpose(glm.mat4(*mat4_np.flatten()))
    chunk_tra_matrix =  glm.translate(glm.mat4(1.0), glm.vec3(*chunk_transl))
    chunk_sca_matrix =  glm.scale(glm.mat4(1.0),  glm.vec3(chunk_scal))
    return chunk_tra_matrix* chunk_sca_matrix* chunk_rot_matrix,chunk_tra_matrix*  chunk_rot_matrix




def compute_camera_matrix(chunk,id_camera):
    TSR,TR = chunk_matrix(chunk)
    cf = glm.transpose(glm.mat4(*chunk.cameras[id_camera].transform))
    center = TSR * cf[3]
    camera_frame = TR * cf
    camera_frame[3][0] = center.x
    camera_frame[3][1] = center.y
    camera_frame[3][2] = center.z
    camera_frame[3][3] = 1.0
    camera_matrix = glm.inverse(camera_frame)
    return camera_matrix,camera_frame

def display_chunk( chunk,tb):
    global show_image
    global user_matrix
    global projection_matrix
    global view_matrix
    global user_camera
    global id_camera
    global texture_IMG_id
    global vao_fsq
    global project_image
    global W
    global H


    
   

    glBindFramebuffer(GL_FRAMEBUFFER,fbo_ids.id_fbo)
    glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
    
    glViewport(0,0,W,H)

    glClearBufferfv(GL_COLOR, 0, [0.0,0.2,0.23,1.0])  # attachment 1
    glClearBufferfv(GL_DEPTH, 0, [1.0])

    glUseProgram(shader0.program)

    cm = chunk_matrix(chunk)[0]
    glUniformMatrix4fv(shader0.uni("uChunk"),1,GL_FALSE, glm.value_ptr(cm))

    if(user_camera):
        # a view of the scene
        view_matrix = user_matrix
        projection_matrix = glm.perspective(glm.radians(45), 1.5,0.1,10)
        glUniformMatrix4fv(shader0.uni("uProj"),1,GL_FALSE, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(shader0.uni("uTrack"), 1, GL_FALSE, glm.value_ptr(tb_matrix := tb.matrix()))
    else:
        # the view from the current_camera_id
        view_matrix = compute_camera_matrix(chunk,id_camera)[0]
        
    glUniformMatrix4fv(shader0.uni("uView"),1,GL_FALSE,  glm.value_ptr(view_matrix))
    glUniform1i(shader0.uni("uMode"),user_camera)
    glUniform1i(shader0.uni("uModeProj"),project_image)

    set_sensor(shader0,chunk.sensors[chunk.cameras[id_camera].sensor_id])

    glActiveTexture(GL_TEXTURE0)
    for model in chunk.models:
        if model.enabled:
            r = model.renderable
            glBindTexture(GL_TEXTURE_2D, r.texture_id)
            if(project_image):
                # texture the geometry with the current id_camera image
                glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
                glUniformMatrix4fv(shader0.uni("uViewCam"),1,GL_FALSE,  glm.value_ptr(camera_matrix))
            else:
                # use the texture of the mesh
                glBindTexture(GL_TEXTURE_2D, r.texture_id)

            #draw the geometry
            glUniform1i(shader0.uni("uUseColor"), False)  #
            glUniform1f(shader0.uni("uClickableId"),0)
            glBindVertexArray( r.vao )
            glDrawArrays(GL_TRIANGLES, 0, r.n_faces*3  )
            glBindVertexArray( 0 )
   
    #  draw the camera frames
    if(user_camera):
        glUseProgram(shader_frame.program)
        glUniformMatrix4fv(shader_frame.uni("uProj"),1,GL_FALSE, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(shader_frame.uni("uTrack"), 1, GL_FALSE, glm.value_ptr(tb_matrix := tb.matrix()))
        glUniformMatrix4fv(shader_frame.uni("uView"),1,GL_FALSE,  glm.value_ptr(view_matrix))
       
        for i in range(0,len(chunk.cameras)):
             # if(i == id_camera):
               # camera_frame = chunk_matrix * ((glm.transpose(glm.mat4(*cameras[i].transform))))
            if chunk.cameras[i].enabled:
                if highligthed_camera_id == i+1:
                    glUniform1f(shader_frame.uni("uScale"), chunk.diagonal*0.06)
                else:
                    glUniform1f(shader_frame.uni("uScale"), chunk.diagonal*0.02)

                camera_frame = compute_camera_matrix(chunk,i)[1]
                track_mul_frame = tb.matrix()*camera_frame
                glUniformMatrix4fv(shader_frame.uni("uTrack"),1,GL_FALSE, glm.value_ptr(track_mul_frame))

                glBindVertexArray(vao_frame )
                glDrawArrays(GL_LINES, 0, 6)                    
                glBindVertexArray( 0 )
        glUseProgram(0)

        # draw the clickable areas
        glUseProgram(shader_clickable.program)
        glDrawBuffers(1, [GL_COLOR_ATTACHMENT1])
        glClearBufferfv(GL_COLOR, 0, [0,0.0,0.0,1.0])  # attachment 

        for i in range(0,len(chunk.cameras)):
            if chunk.cameras[i].enabled:
                _,camera_frame = compute_camera_matrix(chunk,i)

                #draw the invisible clicakble
                camera_center = glm.vec4(camera_frame[3])
                camera_center = projection_matrix*view_matrix*tb.matrix() *glm.vec4(camera_frame[3])
                camera_center /= camera_center.w

                glUniform1f(shader_clickable.uni("uClickableId"), float(i+1) )
                glUniform1f(shader_clickable.uni("uSca"), 1.0/W*10.0)
                camera_center = glm.vec2(camera_center.x,camera_center.y) 
                glUniform2fv(shader_clickable.uni("uTra"), 1, glm.value_ptr(camera_center))

                glBindVertexArray(vao_fsq )
                glDrawArrays(GL_TRIANGLES, 0, 6)                    
                glBindVertexArray( 0 )

        glUseProgram(0)

    if(user_camera == 0 and show_image ):
        #draw the image as a full screen
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glUseProgram(shader_fsq.program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        glUniform1i(shader_fsq.uni("uColorTex"),0)

        glBindVertexArray(vao_fsq )
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0 )

        glUseProgram(0)
        glDisable(GL_BLEND)    

    glBindFramebuffer(GL_FRAMEBUFFER,0)

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_ids.id_fbo)
    #glReadBuffer(GL_COLOR_ATTACHMENT1)  
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

    glBlitFramebuffer(
        0, 0, W, H,
        0, 0, W, H,
        GL_COLOR_BUFFER_BIT,
        GL_LINEAR
    )



def clicked(x,y):
    global tb 
    y = viewport[3] - y
    depth = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT).item()
    mm = np.array(view_matrix*tb.matrix(), dtype=np.float64).flatten()
    pm = np.array(projection_matrix, dtype=np.float64).flatten()
    p  =gluUnProject(x,y,depth, mm,pm, np.array(viewport, dtype=np.int32))

    p_NDC = np.array([x/float(viewport[2])*2.0-1.0,y/float(viewport[3])*2.0-1.0,-1.0+2.0*depth], dtype=np.float64)
    p_mm = glm.inverse(projection_matrix) * glm.vec4(p_NDC[0],p_NDC[1],p_NDC[2], 1.0)
    p_mm /= p_mm.w
    p_w = glm.inverse(view_matrix) * p_mm
    p_w /= p_w.w
    p = p_w
    return p, depth

def get_id(x, y):
    y = viewport[3] - y
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_ids.id_fbo)
    glReadBuffer(GL_COLOR_ATTACHMENT1)
    id = glReadPixels(x, y, 1, 1, GL_RED, GL_FLOAT)
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return id
    
def load_camera_image( id):
    global id_loaded
    global texture_IMG_id
    filename =   imgs_path +"/"+ cameras[id].label+".JPG" 
    print(f"loading {cameras[id].label}.JPG")
    glDeleteTextures(1, [texture_IMG_id])
    texture_IMG_id,_,__ = texture.load_texture(filename)
    id_loaded = id

def load_mesh(filename, textures=[]):
    global ms
    # Load the mesh using PyMeshLab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    mesh = ms.current_mesh()

    # Extract vertices, faces, and texture coordinates
    vertices = mesh.vertex_matrix()

    faces = mesh.face_matrix()
    wed_tcoord = mesh.wedge_tex_coord_matrix()
    if( mesh.has_wedge_tex_coord()):
         ms.apply_filter("compute_texcoord_transfer_wedge_to_vertex")

    texture_id = -1
    if mesh.textures():
        texture_dict = mesh.textures()
        texture_name = next(iter(texture_dict.keys()))  # Get the first key    
        texture_name = os.path.join(os.path.dirname(filename), os.path.basename(texture_name))
        texture_id,w,h = texture.load_texture(texture_name)
    else:
        texture_name = os.path.join(os.path.dirname(filename), textures[0])
        texture_id,w,h = texture.load_texture(texture_name)

    #texture_path = os.path.join(os.path.dirname(filename), os.path.basename(texture_name))
    #imgdata = Image.open(texture_path)
   # maskout.domain_mask =  np.flipud(np.array(imgdata, dtype=np.uint8))

    # Compute the bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    print(f"Bounding Box Min: {bbox_min}")
    print(f"Bounding Box Max: {bbox_max}")

    # Compute the diagonal length of the bounding box
    diagonal_length = ((bbox_max - bbox_min) ** 2).sum() ** 0.5
    print(f"Bounding Box Diagonal Length: {diagonal_length}")

    print(f"vertices: {len(vertices) }")
    print(f"faces: {len(faces)}")

    return vertices, faces, wed_tcoord, bbox_min,bbox_max,texture_id, w,h

def load_model(mod):
    temp_dir, extracted = zip_utils.extract_paths_to_tempdir(msd.file_path, [mod.mesh_path]+ mod.textures )
    current_dir = os.getcwd()
    os.chdir(temp_dir)

    
    vertices, faces, wed_tcoord, bbox_min,bbox_max,texture_id, w,h = load_mesh(mod.mesh_path,mod.textures)
    mod.renderable = renderable(vao=create_buffers(vertices,wed_tcoord,faces),n_verts=len(vertices),n_faces=len(faces),texture_id=texture_id)
    mod.bbox_min = bbox_min
    mod.bbox_max = bbox_max

    os.chdir("..")
    zip_utils.rmdir_if_exists(temp_dir)
    os.chdir(current_dir)


def load_models():
    global msd
    for chunk in msd.chunks:
        for model in chunk.models:
            load_model(model)

def reset_display_image():
    global mask_zoom
    global mask_xpos
    global mask_ypos
    global mouseX
    global mouseY
    global curr_zoom
    global curr_center
    global is_translating   
    global tra_xstart
    global tra_ystart
    global curr_tra
    global show_mask
    global show_image


    show_mask = False
    show_image = False
    mask_zoom = 1.0
    mask_xpos = W/2
    mask_ypos = H/2
    mouseX = W/2
    mouseY = H/2
    curr_zoom   = 1.0
    curr_center = glm.vec2(0.0, 0.0)
    is_translating  = False   
    tra_xstart  = mouseX 
    tra_ystart = mouseX
    curr_tra = glm.vec2(0.0, 0.0)
    show_mask = False

def set_sensor(shader,sensor):
    glUniform1i(shader.uni("uMasks"),3)
    glUniform1i(shader.uni("resolution_width"),sensor.resolution["width"])
    glUniform1i(shader.uni("resolution_height"),sensor.resolution["height"])
    glUniform1f(shader.uni("f" ) ,sensor.calibration["f"]) 
    glUniform1f(shader.uni("cx"),sensor.calibration["cx"])
    glUniform1f(shader.uni("cy"),-sensor.calibration["cy"])
    glUniform1f(shader.uni("k1"),sensor.calibration["k1"])
    glUniform1f(shader.uni("k2"),sensor.calibration["k2"])
    glUniform1f(shader.uni("k3"),sensor.calibration["k3"])
    glUniform1f(shader.uni("p1"),sensor.calibration["p1"])
    glUniform1f(shader.uni("p2"),sensor.calibration["p2"])
   


def draw_xml_node(node):
            flags = 0
            opened = imgui.tree_node(node.tag, flags)

            if opened:
                for k, v in node.attrib.items():
                    imgui.text(f'{k}="{v}"')

                if node.text and node.text.strip():
                    imgui.text(node.text.strip())
                for child in node:
                    draw_xml_node(child)

                imgui.tree_pop()

def draw_xml_modal(root):
    stay_open = True
    imgui.open_popup("XML Viewer")

    if imgui.begin_popup_modal(
        "XML Viewer",
        flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE
    )[0]:

        imgui.text("XML Viewer")
        imgui.separator()

        imgui.begin_child("xml_scroll", 600, 400, True)
        draw_xml_node(root)
        imgui.end_child()

        imgui.separator()

        if imgui.button("Close"):
            imgui.close_current_popup()
            stay_open = False

        imgui.end_popup()
        return stay_open

def  show_xml(metashape_file):
    global metashape_root
    with zipfile.ZipFile(metashape_file, 'r') as zip_ref:
        with zip_ref.open("doc.xml") as file:
        # Parse the XML file
            tree = ET.parse(file)
            metashape_root = tree.getroot()
           # draw_xml_node(metashape_root)
            draw_xml_modal(metashape_root)

checkbox_state = {}

def draw_metashape_structure():

    for chunk in msd.chunks:
        chunk_id = chunk.id
        key = f"chunk{chunk_id}"
        if key not in checkbox_state:
            checkbox_state[key] = True
        _, checkbox_state[key] = imgui.checkbox(key, checkbox_state[key])
 
        imgui.same_line()   
        if imgui.tree_node(f"{chunk_id}"):
            # Cameras
            if imgui.tree_node("Cameras"):
                for cam in chunk.cameras:
                    key = f"chunk{chunk_id}_camera_{cam.label}"
                    if key not in checkbox_state:
                        checkbox_state[key] = cam.enabled
                    _, checkbox_state[key] = imgui.checkbox(cam.label, checkbox_state[key])
                    if _:
                        cam.enabled = not cam.enabled   
                imgui.tree_pop()

            # Models
            if imgui.tree_node("Models"):
                for model in chunk.models:
                    key = f"chunk{chunk_id}_model_{model.mesh_path}"
                    if key not in checkbox_state:
                        checkbox_state[key] = model.enabled
                    _, checkbox_state[key] = imgui.checkbox(model.mesh_path, checkbox_state[key])
                    if _:
                        model.enabled = not model.enabled   
                imgui.tree_pop()

            imgui.tree_pop()

    # Display the images path
    imgui.text(f"Images Path: {msd.images_path}")
    if imgui.button("Browse##images_folder"):
        folder = filedialog.askdirectory(title="Select Images Folder")
        if folder:
            msd.images_path = folder

def set_view(chunk,mod):
    global viewport
    global user_matrix
    global projection_matrix
    clock = pygame.time.Clock()
    viewport =[0,0,W,H]

        # cm = chunk_matrix(chunk)[0]
        # bbmin = cm * glm.vec4(glm.vec3(mod.bbox_min), 1.0)
        # bbmax = cm * glm.vec4(glm.vec3(mod.bbox_max), 1.0)
        # center = glm.vec3((bbmin + bbmax)) / 2.0
        # chunk.diagonal = glm.length(bbmax - bbmin)


    cd = chunk.diagonal
    center = chunk.center

    eye = center + glm.vec3(2*cd,0,0)
    user_matrix = glm.lookAt(glm.vec3(eye),glm.vec3(center), glm.vec3(0,0,1)) # TODO: UP PARAMETRICO !
    projection_matrix = glm.perspective(glm.radians(45),W/float(H),cd*0.1,cd*2) # TODO: NEAR E FAR PARAMETRICI !!
    tb.set_center_radius(center, cd)

def compute_chunks_bbox(msd):
    for chunk in msd.chunks:
        # compute the bounding box of the chunk
        bmin = glm.vec3(1e10,1e10,1e10)
        bmax = glm.vec3(-1e10,-1e10,-1e10)
        chunk.center = None
        cm = chunk_matrix(chunk)[0]
        for model in chunk.models:
            bbmin = cm * glm.vec4(glm.vec3(model.bbox_min), 1.0)
            bbmax = cm * glm.vec4(glm.vec3(model.bbox_max), 1.0)
            bmin = glm.min(bmin, glm.vec3(bbmin))
            bmax = glm.max(bmax, glm.vec3(bbmax))
            
        if bmax.x > bmin.x:
            chunk.center = (bmin + bmax) / 2.0
            chunk.diagonal = glm.length(bmax - bmin)
            continue

        for camera in chunk.cameras:
            cf = glm.transpose(glm.mat4(*camera.transform))
            center = cm * cf[3]
            bmin = glm.min(bmin, glm.vec3(center))
            bmax = glm.max(bmax, glm.vec3(center))

        chunk.center = (bmin + bmax) / 2.0
        chunk.diagonal = glm.length(bmax - bmin)

def main():
    glm.silence(4)
    global W
    global H
    W = 1200
    H = 800

    global tb

    global vao_frame
    global shader_fsq
    global shader_clickable
    global texture_IMG_id
    global show_image
    global vao_fsq
    global id_loaded
    global project_image
    global shader0
    global user_matrix
    global app_path
    
    global mask_zoom
    global mask_xpos
    global mask_ypos
    global mouseX
    global mouseY
    global curr_zoom
    global curr_center
    global is_translating   
    global tra_xstart
    global tra_ystart
    global curr_tra
    global quadric

    global metashape_root
    global msd 
    global fbo_ids

    global highligthed_camera_id
    highligthed_camera_id = 0

    msd = None

    is_translating = False

    np.random.seed(42)  # For reproducibility



    id_loaded = -1
    show_image = False
    project_image = False

    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.OPENGL|pygame.DOUBLEBUF)
    pygame.display.set_caption("Metashape viewer")
  
    max_ssbo_size = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
    print(f"Max SSBO size: {max_ssbo_size / (1024*1024):.2f} MB")
    max_texture_units = glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)
    print(f"Max texture units: {max_texture_units}")

    max_compute_texture_units = glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS)
    print(f"Max compute shader texture image units: {max_compute_texture_units}")

    # Initialize ImGui
    imgui.create_context()
    imgui_renderer = PygameRenderer()

    # Set ImGui's display size to match the window size
    imgui.get_io().display_size = (W,H)  # Ensure valid display size

    quadric = gluNewQuadric()

    fbo_ids = fbo.fbo(W,H)
    

    tb = trackball.Trackball()
    tb.reset()
 
    glEnable(GL_DEPTH_TEST)
    
    app_path = os.path.dirname(os.path.abspath(__file__))
    print(f"App path: {app_path}")


    #os.chdir(main_path)
    #vertices, faces, wed_tcoords, bmin,bmax,texture_id,texture_w,texture_h  = load_mesh(mesh_name) 
 
        

    global shader0
    global shader_frame

    shader0     = shader(shaders.vertex_shader, shaders.fragment_shader)
    shader_fsq  = shader(shaders.vertex_shader_fsq, shaders.fragment_shader_fsq)
    shader_clickable = shader(shaders.vertex_shader_clickable, shaders.fragment_shader_clickable)
    shader_frame = shader(shaders.vertex_shader_frame, shaders.fragment_shader_frame)
   

    check_gl_errors()

    vao_frame = create_buffers_frame()
    vao_fsq = create_buffers_fsq()

    global viewport
    clock = pygame.time.Clock()



    global id_camera
    global user_camera
    id_camera   = 0

    user_camera = 1
    texture_IMG_id = 0
    mask_zoom = 1.0
    mask_xpos = W/2
    mask_ypos = H/2
    curr_zoom = 1.0
    curr_center = glm.vec2(0,0)
    tra_xstart = mask_xpos
    tra_ystart = mask_ypos
    curr_tra = glm.vec2(0,0)


    root = Tk()
    root.withdraw()

    global selected_file
    selected_file = None
    user_view_on = False
     

    while True:    
         
        time_delta = clock.tick(60)/1000.0 
        for event in pygame.event.get():
            imgui_renderer.process_event(event)
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                return  
            if event.type == pygame.MOUSEMOTION:
                mouseX, mouseY = event.pos
                
                if user_view_on:
                    highligthed_camera_id = get_id(mouseX, mouseY)
                    
                if show_image and is_translating:
                    mask_xpos = mouseX
                    mask_ypos = mouseY
                else:    
                    if user_view_on: tb.mouse_move(projection_matrix, user_matrix, mouseX, mouseY)

            if event.type == pygame.MOUSEWHEEL:
                xoffset, yoffset = event.x, event.y
                if show_image:
                    mask_zoom = 1.1 if yoffset > 0 else 0.97
                    if yoffset > 0 :
                        mask_xpos = mouseX 
                        mask_ypos = mouseY
                else:
                    if user_view_on: tb.mouse_scroll(xoffset, yoffset)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button not in (4, 5):
                    mouseX, mouseY = event.pos
                    keys = pygame.key.get_pressed()  # Get the state of all keys
                    if show_image:
                        is_translating = True
                        tra_xstart = mouseX
                        tra_ystart = mouseY
                        mask_xpos =  mouseX
                        mask_ypos =  mouseY
                    else:
                        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:  
                            cp,depth = clicked(mouseX,mouseY)
                            if depth < 0.99:
                                if user_view_on: tb.reset_center(cp)               
                        else:
                            if user_view_on: tb.mouse_press(projection_matrix, user_matrix, mouseX, mouseY)
            if event.type == pygame.MOUSEBUTTONUP:
                mouseX, mouseY  = event.pos
                if event.button == 1:  # Left mouse button
                    if show_image:
                        is_translating = False
                    else:
                        if user_view_on: tb.mouse_release()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    user_camera = 1 - user_camera

        imgui.new_frame()

        if imgui.begin_main_menu_bar():

            # first menu dropdown
            if imgui.begin_menu('Files', True):
                clicked_open, _ = imgui.menu_item("Open Metashape", "Ctrl+O", False, True)
                if clicked_open:
                    selected_file = filedialog.askopenfilename(
                        title="Open Metashape file",
                        filetypes=[
                            ("Metashape PSZ", "*.psz"),
                            ("All files", "*.*"),
                        ]
                    )
                    print("Selected:", selected_file)
                    if selected_file:
                        
                        msd = metashape_loader.load_psz(selected_file)
                        msd.file_path = selected_file
                        # Check if images exist, if not ask for folder
                        for chunk in msd.chunks:
                            if len(chunk.cameras) > 0:
                                filename =   msd.images_path +"/"+ chunk.cameras[0].label+".JPG" 
                                if not os.path.exists(filename):
                                    folder = filedialog.askdirectory(title="Select Images Folder")
                                    if folder:
                                        msd.images_path = folder
                        load_models()
                        compute_chunks_bbox(msd)

                        #show_xml(metashape_file)
                        #show_load_gui = True
                        selected_file = None

                imgui.end_menu()

           # if show_load_gui:
           #     show_load_gui = draw_xml_modal(metashape_root)

            if  imgui.begin_menu('data', True):
                if msd is not None:
                    draw_metashape_structure()
                imgui.end_menu()

            imgui.end_main_menu_bar()


            

    
        check_gl_errors()

        if msd is not None:
            user_view_on = True
            if show_image:
                display_image( msd.chunks[1])
            else:
                set_view(msd.chunks[1],msd.chunks[1].models[0])
                for chunk in msd.chunks:
                    display_chunk( chunk,tb) 
        
        #display(shader0, rend,tb)
        glBindVertexArray( 0 )
        check_gl_errors()


        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())

        # Draw the UI elements
        pygame.display.flip()

        clock = pygame.time.Clock()


if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()
