from torch import chunk
import labelling as lb
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



def create_buffers_camera():
    
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
    
    verts = [0,0,0,  1,-1, 1,  1, 1, 1,
             0,0,0,  1, 1, 1,  -1, 1, 1, 
             0,0,0, -1, 1, 1,  -1, -1, 1, 
             0,0,0, -1,-1, 1,   1, -1, 1 
             ]
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
    
    col = [0.8,0,0, 0.8,0,0, 0.8,0,0, 
           0,0.8,0, 0,0.8,0, 0,0.8,0, 
           0.8,0,0, 0.8,0,0, 0.8,0,0, 
           1,1,1, 1,1,1, 1,1,1 
           ]
    col = np.array(col, dtype=np.float32)

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,col.nbytes, col, GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    glBindVertexArray( 0 )
    
    # Unbind other stuff
    glDisableVertexAttribArray(position)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vertex_array_object


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




def display_image(chunk,id_camera):
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
        global shader_basic
        

        sensor = chunk.sensors[chunk.cameras[id_camera].sensor_id]
        
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

        glBindVertexArray(vao_fsq )
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0 )


        glActiveTexture(current_unit)
        glBindTexture(GL_TEXTURE_2D, current_texture)
        glUseProgram(0)

       #labelling
        #return
        glUseProgram(shader_basic.program)

        dx = 8.0/W
        dy = dx *W/H 
        for i,p2d in enumerate(chunk.cameras[id_camera].projecting_samples_pos):
            px = p2d[0]/float(sensor.resolution["width"])  * 2.0 - 1.0  
            py = p2d[1]/float(sensor.resolution["height"]) * 2.0 - 1.0  
            glUniform2f(shader_basic.uni("uPos"),0.0,0.0)

            px = px * curr_zoom + tra.x
            py = py * curr_zoom + tra.y

            g_i = chunk.cameras[id_camera].projecting_samples_ids[i]
            label_i = lb.sample_points[g_i].label

            c = [0,0,0]
            if  label_i is not None:
                c = lb.labels[  label_i ].color

            glUniform3f(shader_basic.uni("uColor"),c[0],c[1],c[2])
            glBegin(GL_LINE_STRIP)
            glVertex3f(px-dx, py-dy , -0.1)
            glVertex3f(px+dx, py-dy , -0.1)
            glVertex3f(px+dx, py+dy , -0.1)
            glVertex3f(px-dx, py+dy , -0.1)
            glVertex3f(px-dx, py-dy , -0.1)
            glEnd()

            glUniform3f(shader_basic.uni("uColor"),1,1,1)
            if i == curr_sel_sample_id:
                glBegin(GL_LINE_STRIP)
                glVertex3f(px-dx*1.3, py-dy*1.3 , -0.1)
                glVertex3f(px+dx*1.3, py-dy*1.3 , -0.1)
                glVertex3f(px+dx*1.3, py+dy*1.3 , -0.1)
                glVertex3f(px-dx*1.3, py+dy*1.3 , -0.1)
                glVertex3f(px-dx*1.3, py-dy*1.3 , -0.1)
                glEnd()



            
        
        glUseProgram(0)

def get_selected_sample(chunk,id_camera,x,y):
    sensor = chunk.sensors[chunk.cameras[id_camera].sensor_id]    
    x = (x / float(W))*2.0-1.0 
    y = (1.0 - y / float(H))*2.0-1.0 

    for i,p2d in enumerate(chunk.cameras[id_camera].projecting_samples_pos):
        px = p2d[0]/float(sensor.resolution["width"])  * 2.0 - 1.0  
        py = p2d[1]/float(sensor.resolution["height"]) * 2.0 - 1.0  

        px = px * curr_zoom + curr_center.x + curr_tra.x
        py = py * curr_zoom + curr_center.y + curr_tra.y

        dist =  glm.length(glm.vec2(px,py)-glm.vec2(x,y))

        if dist < 0.01:
            return i

    return -1

           

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


def project_point(sensor, p):
    k1 = sensor.calibration["k1"]
    k2 = sensor.calibration["k2"]
    k3 = sensor.calibration["k3"]
    k4 = sensor.calibration["k4"]
    p1 = sensor.calibration["p1"]
    p2 = sensor.calibration["p2"]
    f = sensor.calibration ["f"]
    cx = sensor.calibration ["cx"]
    cy = -sensor.calibration ["cy"]
    b1 = sensor.calibration ["b1"]
    b2 = sensor.calibration ["b2"]
    resolution_width = sensor.calibration["resolution"]["width"]
    resolution_height = sensor.calibration["resolution"]["height"]

    x = p.x/p.z
    y = -p.y/p.z
    r = glm.sqrt(x*x+y*y)
    r2 = r*r
    r4 = r2*r2
    r6 = r4*r2
    r8 = r6*r2

    A = (1.0 + k1*r2+k2*r4+k3*r6 + k4*r8 )
    B = (1.0 )

    xp = x * A+ (p1*(r2+2*x*x)+2*p2*x*y) * B
    yp = y * A+ (p2*(r2+2*y*y)+2*p1*x*y) * B

    pix_i = resolution_width*0.5+cx+xp*f+xp*b1+yp*b2
    pix_j = resolution_height*0.5+cy+yp*f

    return round(pix_i), round(pix_j)

def project_point_to_camera(chunk,camera_id, p):
    global curr_camera_depth
    camera = chunk.cameras[camera_id]
    sensor = chunk.sensors[camera.sensor_id]
    near = camera.near
    far  = camera.far

    cm, _ = compute_camera_matrix(chunk,camera_id)

    p_cam = cm * glm.vec4(p[0], p[1], p[2], 1.0)
    pix_i, pix_j = project_point(sensor, glm.vec3(p_cam.x, p_cam.y, p_cam.z))
    if pix_i >=0 and pix_i < sensor.resolution["width"] and  pix_j >=0 and pix_j < sensor.resolution["height"]:#frustum
        pix_z = (p_cam.z - near)/(far-near)
        comp_z = curr_camera_depth[pix_j][pix_i] #depth test
        if pix_z < comp_z+0.0001:
            return pix_i, pix_j

    return -1,-1

def compute_near_far_for_camera(chunk,camera_id, points):
    global curr_camera_depth
    camera = chunk.cameras[camera_id]
    sensor = chunk.sensors[camera.sensor_id]

    cm, _ = compute_camera_matrix(chunk,camera_id)

    near = np.finfo(np.float64).max
    far = 0
    for p in points[::20]:
        p_cam = cm * glm.vec4(p[0], p[1], p[2], 1.0)
        pix_i, pix_j = project_point(sensor, glm.vec3(p_cam.x, p_cam.y, p_cam.z))
        if pix_i >=0 and pix_i < sensor.resolution["width"] and  pix_j >=0 and pix_j < sensor.resolution["height"]:#frustum
           if p_cam.z > 0:
               near = min ( near, p_cam.z)
               far  = max ( far, p_cam.z) 

    camera = chunk.cameras[camera_id].near = near
    camera = chunk.cameras[camera_id].far = far


def project_samples_to_camera(chunk, camera_id, samples):
    for i, sp in enumerate(samples):
        if [chunk.id,camera_id] not in sp.camera_refs:

            pix_i, pix_j = project_point_to_camera(chunk, camera_id,sp.position)
            if pix_i > 0:
                sp.camera_refs.append([chunk.id,camera_id])
                sp.projected_coords.append([pix_i,pix_j])
                chunk.cameras[camera_id].projecting_samples_ids.append(i)
                chunk.cameras[camera_id].projecting_samples_pos.append([pix_i,pix_j])



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



import imageio.v2 as imageio
def compute_camera_depth(chunk, id_camera):
    global fbo_camera

    sensor = chunk.sensors[chunk.cameras[id_camera].sensor_id]
    fbo_camera = fbo.fbo(sensor.resolution["width"],sensor.resolution["height"])

    glBindFramebuffer(GL_FRAMEBUFFER,fbo_camera.id_fbo)
    glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
    
    glViewport(0,0,sensor.resolution["width"],sensor.resolution["height"])
    
    glClearBufferfv(GL_COLOR, 0, [0.0,0.2,0.23,1.0])  # attachment 1
    glClearBufferfv(GL_DEPTH, 0, [1])

    view_matrix = compute_camera_matrix(chunk,id_camera)[0]

    glUseProgram(shader0.program)
    cm = chunk_matrix(chunk)[0]
    glUniformMatrix4fv(shader0.uni("uChunk"),1,GL_FALSE, glm.value_ptr(cm))
    glUniformMatrix4fv(shader0.uni("uView"),1,GL_FALSE,  glm.value_ptr(view_matrix))
    glUniform1i(shader0.uni("uMode"),False)
    glUniform1i(shader0.uni("uModeProj"),False)
    set_sensor(shader0,chunk.sensors[chunk.cameras[id_camera].sensor_id],chunk.cameras[id_camera].near,chunk.cameras[id_camera].far )


    glUniform1i(shader0.uni("uUseColor"), True)  #
    col = glm.vec3(1.0,0.0,0.0)
    glUniform3fv(shader0.uni("uColor"), 1, glm.value_ptr(col))

    for model in chunk.models:
            if model.enabled:
                r = model.renderable
                glBindVertexArray( r.vao )
                glDrawArrays(GL_TRIANGLES, 0, r.n_faces*3  )
                glBindVertexArray( 0 )

    glUseProgram(0)

    data = glReadPixels(0, 0, sensor.resolution["width"], sensor.resolution["height"], GL_DEPTH_COMPONENT, GL_FLOAT)

    depth_buffer = np.frombuffer(data, dtype=np.float32)
    min_depth = depth_buffer.min()
    max_depth = depth_buffer.max()

    print(f" range {min_depth}-{max_depth}")

    depth_buffer = depth_buffer.reshape((sensor.resolution["height"], sensor.resolution["width"]))

    glBindFramebuffer(GL_FRAMEBUFFER,0)


    return depth_buffer




def display_chunk( chunk,tb):
    global show_image
    global user_matrix
    global projection_matrix
    global view_matrix
    global user_camera
    global id_camera
    global texture_IMG_id
    global vao_fsq
    global vao_camera
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
        #projection_matrix = glm.perspective(glm.radians(45), 1.5,0.1,10)
        glUniformMatrix4fv(shader0.uni("uProj"),1,GL_FALSE, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(shader0.uni("uTrack"), 1, GL_FALSE, glm.value_ptr(tb_matrix := tb.matrix()))
    else:
        # the view from the current_camera_id
        if chunk.cameras[id_camera].near == None:
            compute_near_far_for_camera(chunk,id_camera,chunk.models[0].verts)
        set_sensor(shader0,chunk.sensors[chunk.cameras[id_camera].sensor_id],chunk.cameras[id_camera].near,chunk.cameras[id_camera].far)
        view_matrix = compute_camera_matrix(chunk,id_camera)[0]
        
    glUniformMatrix4fv(shader0.uni("uView"),1,GL_FALSE,  glm.value_ptr(view_matrix))
    glUniform1i(shader0.uni("uMode"),user_camera)
    glUniform1i(shader0.uni("uModeProj"),project_image)
    glUniform1i(shader0.uni("uColorMode"), 2)  #


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
            glBindVertexArray( r.vao )
            glDrawArrays(GL_TRIANGLES, 0, r.n_faces*3  )
            #glDrawArrays(GL_POINTS, 0, r.n_faces*3  )
            glBindVertexArray( 0 )




    #draw the sample points in worldspace 3D
    #glUniform1i(shader0.uni("uColorMode"), 1)  #
     
    #if user_camera:
    ##    for p in lb.sample_points:
    #        if p.label != None:
    #            col = lb.labels[p.label].color
    #        else:
    #            col = [0,0,0]
    #        glUniform3fv(shader0.uni("uColor"), 1,  col)
    #        model_matrix = glm.translate(glm.mat4(1.0), p.position)
    #        glUniformMatrix4fv(shader0.uni("uChunk"),1,GL_FALSE, glm.value_ptr(model_matrix))
    #        gluSphere(quadric,0.0005,4,4)
        

    glUseProgram(0)



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
                    glUniform1f(shader_frame.uni("uScale"), chunk.diagonal*0.006)
                else:
                    glUniform1f(shader_frame.uni("uScale"), chunk.diagonal*0.002)

                camera_frame = compute_camera_matrix(chunk,i)[1]
                track_mul_frame = tb.matrix()*camera_frame*glm.scale(glm.mat4(1),glm.vec3(1,1,2))
                glUniformMatrix4fv(shader_frame.uni("uTrack"),1,GL_FALSE, glm.value_ptr(track_mul_frame))

                #glBindVertexArray(vao_frame )
                #glDrawArrays(GL_LINES, 0, 6)                    
                #glBindVertexArray( 0 )

                glBindVertexArray(vao_camera )
                glDrawArrays(GL_TRIANGLES, 0, 12)                    
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
                glUniform1f(shader_clickable.uni("uSca"), 1.0/W*20.0)
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
    
def load_camera_image( chunk,id):
    global id_loaded
    global texture_IMG_id
    filename =   msd.images_path +"/"+ chunk.cameras[id].label+".JPG" 
    print(f"loading {chunk.cameras[id].label}.JPG")
    glDeleteTextures(1, [texture_IMG_id])
    texture_IMG_id,_,__ = texture.load_texture(filename)
    id_loaded = id

def load_mesh(filename, textures=[]):
    global ms
    global mesh
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
    mod.verts = vertices

    os.chdir("..")
    zip_utils.rmdir_if_exists(temp_dir)
    os.chdir(current_dir)


def load_models(generate_samples ):
    global msd
    global ms
     
    for chunk in msd.chunks:
        for model in chunk.models:
            load_model(model)

            #LABELLER
            if generate_samples:
                ms.apply_filter("generate_sampling_poisson_disk", radius= pymeshlab.PercentageValue(0.7))
                print(f"mn {ms.mesh_number()}")
                ms.set_current_mesh(1)
                mesh = ms.current_mesh()
                for v in mesh.vertex_matrix():
                    pos_ws = chunk_matrix(chunk)[0] * glm.vec4(v[0], v[1], v[2], 1.0)
                    lb.sample_points.append(lb.SamplePoint(glm.vec3(pos_ws))) 

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

def set_sensor(shader,sensor,near,far):
    glUniform1i(shader.uni("uMasks"),3)
    glUniform1i(shader.uni("resolution_width"),sensor.resolution["width"])
    glUniform1i(shader.uni("resolution_height"),sensor.resolution["height"])
    glUniform1f(shader.uni("f" ) ,sensor.calibration["f"]) 
    glUniform1f(shader.uni("cx"),sensor.calibration["cx"])
    glUniform1f(shader.uni("cy"),-sensor.calibration["cy"])
    glUniform1f(shader.uni("k1"),sensor.calibration["k1"])
    glUniform1f(shader.uni("k2"),sensor.calibration["k2"])
    glUniform1f(shader.uni("k3"),sensor.calibration["k3"])
    glUniform1f(shader.uni("k4"),sensor.calibration["k4"])
    glUniform1f(shader.uni("p1"),sensor.calibration["p1"])
    glUniform1f(shader.uni("p2"),sensor.calibration["p2"])
    glUniform1f(shader.uni("b1"),sensor.calibration["b1"])
    glUniform1f(shader.uni("b2"),sensor.calibration["b2"])
    glUniform1f(shader.uni("uNear"),near)
    glUniform1f(shader.uni("uFar"),far)
    
   


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
            checkbox_state[key] = chunk.enabled
        _, checkbox_state[key] = imgui.checkbox(key, checkbox_state[key])
        if _:
            chunk.enabled = not chunk.enabled
 
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

    cd = chunk.diagonal
    center = chunk.center

    eye = center + glm.vec3(2*cd,0,0)
    user_matrix = glm.lookAt(glm.vec3(eye),glm.vec3(center), glm.vec3(0,0,1))  
    projection_matrix = glm.perspective(glm.radians(45),W/float(H),cd*0.1,cd*4)  
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


def draw_labels(selected_index):
    ROW_HEIGHT = 20
    MAX_VISIBLE_HEIGHT = 700  # height of the scrollable area
   
     # Estimate content height
    content_height = len(lb.labels) * ROW_HEIGHT + 20
    window_height = min(content_height, MAX_VISIBLE_HEIGHT)

    # Window auto-grows until MAX_HEIGHT
    imgui.set_next_window_size_constraints(
        (200, 0),            # min size
        (1000, MAX_VISIBLE_HEIGHT)   # max size
    )

    imgui.set_next_window_size(300, window_height)

    imgui.begin(
        "Labels",
        False,
        imgui.WINDOW_NO_COLLAPSE
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_SCROLLBAR 
    )

    # Scrollable child window
    imgui.begin_child(
        "LabelsChild",
        width=0,  # take full width
        height=MAX_VISIBLE_HEIGHT,
        border=False
    )

    for i, label in enumerate(lb.labels):
        # Create a unique ID by including the index in the label
        selectable_label = f"##row{i}"

        clicked, _ = imgui.selectable(
            selectable_label,
            selected_index == i,
            imgui.SELECTABLE_ALLOW_ITEM_OVERLAP,
            0,
            ROW_HEIGHT
        )

        if clicked:
            selected_index = i

        # Draw the color box and name on the same line
        imgui.same_line()
        imgui.color_button(
            f"##color{i}",
            label.color[0], label.color[1], label.color[2], 1.0,
            0,
            16, 16
        )

        imgui.same_line()
        imgui.text(label.name)

    imgui.end_child()
    imgui.end()

    return selected_index

def load_and_setup_metashape(selected_file,generate_samples,images_path=None):
        global msd
        msd = metashape_loader.load_psz(selected_file)
        msd.file_path = selected_file
        if images_path != None:
            msd.images_path = images_path
        # Check if images exist, if not ask for folder
        for chunk in msd.chunks:
            if len(chunk.cameras) > 0:
                filename =   msd.images_path +"/"+ chunk.cameras[0].label+".JPG" 
                if not os.path.exists(filename):
                    folder = filedialog.askdirectory(title="Select Images Folder")
                    if folder:
                        msd.images_path = folder
        load_models(generate_samples)
        compute_chunks_bbox(msd)
        set_view(msd.chunks[0], msd.chunks[0].models[0])

def main():
    glm.silence(4)
    global W
    global H
    W = 1200
    H = 800

    global tb

    global vao_frame
    global vao_camera
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
    global curr_camera_depth

    global highligthed_camera_id
    highligthed_camera_id = 0

    global viewport
    viewport =[0,0,W,H]

    msd = None

    is_translating = False

    np.random.seed(42)  # For reproducibility



    id_loaded = -1
    show_image = False
    project_image = False

    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.OPENGL|pygame.DOUBLEBUF)
    pygame.display.set_caption("Labeller")

    icon = pygame.image.load("labeller.png")
    pygame.display.set_icon(icon)
  
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
    global shader_basic

    shader0     = shader(shaders.vertex_shader, shaders.fragment_shader)
    shader_fsq  = shader(shaders.vertex_shader_fsq, shaders.fragment_shader_fsq)
    shader_clickable = shader(shaders.vertex_shader_clickable, shaders.fragment_shader_clickable)
    shader_frame = shader(shaders.vertex_shader_frame, shaders.fragment_shader_frame)
    shader_basic = shader(shaders.vertex_shader_basic, shaders.fragment_shader_basic)

    check_gl_errors()

    vao_camera = create_buffers_camera()
    vao_frame = create_buffers_frame()
    vao_fsq = create_buffers_fsq()

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
    user_camera = False

    current_label = 0

    global curr_sel_sample_id
    curr_sel_sample_id = -1 

    metashape_filename  = None
    labels_filename     = None
    project_path        = None

    while True:    
         
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        time_delta = clock.tick(60)/1000.0 
        for event in pygame.event.get():
            imgui_renderer.process_event(event)
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                user_camera = True 
                show_image = False

            if event.type == pygame.MOUSEMOTION:
                mouseX, mouseY = event.pos
                
                if user_camera:
                    highligthed_camera_id = get_id(mouseX, mouseY)
                    
                if show_image and is_translating:
                    mask_xpos = mouseX
                    mask_ypos = mouseY
                else:    
                    if user_camera: tb.mouse_move(projection_matrix, user_matrix, mouseX, mouseY)

            if event.type == pygame.MOUSEWHEEL:
                xoffset, yoffset = event.x, event.y
                if show_image:
                    mask_zoom = 1.1 if yoffset > 0 else 0.97
                    if yoffset > 0 :
                        mask_xpos = mouseX 
                        mask_ypos = mouseY
                else:
                    if user_camera: tb.mouse_scroll(xoffset, yoffset)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = event.pos
                keys = pygame.key.get_pressed()  # Get the state of all keys
                if event.button == 3 : #right button
                    if show_image:   
                        is_translating = True
                        tra_xstart = mouseX
                        tra_ystart = mouseY
                        mask_xpos =  mouseX
                        mask_ypos =  mouseY
                else:
                    if event.button == 1:#left button
                        if show_image:
                           curr_sel_sample_id = get_selected_sample(chunk,id_camera,mouseX,mouseY)
                           if curr_sel_sample_id != -1:
                               g_i = chunk.cameras[id_camera].projecting_samples_ids[curr_sel_sample_id]
                               lb.sample_points[g_i].label = current_label
                               print(f"selected: {curr_sel_sample_id}")
                        else: 
                            if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:  
                                cp,depth = clicked(mouseX,mouseY)
                                if depth < 0.99:
                                    if user_camera: tb.reset_center(cp)         
                            if keys[pygame.K_LSHIFT]:
                                highligthed_camera_id = get_id(mouseX, mouseY)
                                if highligthed_camera_id >= 1:
                                        id_camera = int(highligthed_camera_id)-1
                                        user_camera = False
                                        #load_camera_image( msd.chunks[1],id_camera)
                                        #reset_display_image()
                            else:
                                if user_camera: tb.mouse_press(projection_matrix, user_matrix, mouseX, mouseY)
 
            if event.type == pygame.MOUSEBUTTONUP:
                mouseX, mouseY  = event.pos
                if event.button == 1:  # Left mouse button
                    if user_camera: tb.mouse_release()
                if event.button == 3:  # Right mouse button
                    if show_image:
                            is_translating = False
                    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    user_camera = 1 - user_camera

        imgui.new_frame()

        if  msd != None and not user_camera:
            imgui.set_next_window_position(20, 20, imgui.ONCE)  # fixed position (optional)

            imgui.begin("Camera",False,imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR)   

            changed, checkbox_value = imgui.checkbox(
                    "Show actual image",
                    show_image
                )
            if changed:
                show_image = checkbox_value
                if show_image:
                    if id_loaded != id_camera:
                        load_camera_image( msd.chunks[0],id_camera)

            if imgui.button("project samples"):
                curr_camera_depth = compute_camera_depth(msd.chunks[0], id_camera)
                project_samples_to_camera(msd.chunks[0], id_camera, lb.sample_points)

            imgui.end()
        
        current_label = draw_labels(current_label)

        if imgui.begin_main_menu_bar():

            # first menu dropdown
            if imgui.begin_menu('File', True):
                
                clicked_open, _ = imgui.menu_item("Open Project", "", False, True)
                if clicked_open:
                    selected_file = filedialog.askopenfilename(
                        title="Open Project File",
                        filetypes=[
                            ("Labeller file", "*.json"),
                            ("All files", "*.*"),
                        ]
                    )
                    if selected_file:
                        metashape_filename, images_path,labels_filename, lb.sample_points = lb.load_labelling(selected_file)
                        load_and_setup_metashape(metashape_filename,False,images_path)
                        lb.load_labels(labels_filename)
                        user_camera = True
                        show_image = False

                        project_path = selected_file
                        selected_file = None

                clicked_save, _ = imgui.menu_item("Save Project", "", False, project_path != None)
                if clicked_save:
                     lb.save_labelling(metashape_filename,msd.images_path,labels_filename,project_path)

                clicked_save, _ = imgui.menu_item("Save Project As ...", "", False, metashape_filename != None and labels_filename != None)
                if clicked_save:
                    new_path = filedialog.asksaveasfilename(
                        title="Save project As ...",
                        defaultextension=".json",
                        filetypes=[
                            ("Labeller json", "*.json"),
                            ("All files", "*.*"),
                        ]
                    )
                    if new_path:
                        project_path = new_path
                        lb.save_labelling(metashape_filename,msd.images_path,labels_filename,project_path)

                imgui.separator() 

                clicked_open, _ = imgui.menu_item("Open Metashape", "", False, True)
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
                        load_and_setup_metashape(selected_file,True)
                        user_camera = True
                        metashape_filename = selected_file
                        selected_file = None

                 
                clicked_open, _ = imgui.menu_item("Open Labels", "", False, True)
                if clicked_open:
                    selected_file = filedialog.askopenfilename(
                        title="Open Metashape file",
                        filetypes=[
                            ("Labels", "*.json"),
                            ("All files", "*.*"),
                        ]
                    )
                    print("Selected:", selected_file)
                    if selected_file:
                        lb.load_labels(selected_file)
                    labels_filename = selected_file
                    selected_file = None
                    
                
                imgui.end_menu()

                    




            if  imgui.begin_menu('data', True):
                if msd is not None:
                    draw_metashape_structure()
                imgui.end_menu()

            imgui.end_main_menu_bar()

    
        check_gl_errors()

        if msd is not None:
            if show_image:
                display_image(msd.chunks[0],id_camera)
            else:
             #   set_view(msd.chunks[1], msd.chunks[1].models[0])
                for chunk in msd.chunks:
                    if chunk.enabled:
                        display_chunk(chunk, tb)


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
