
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
from  renderable import * 
import numpy as np
import os
import ctypes
import numpy as np
import trackball 
import texture
import metashape_loader
import fbo
from  shaders import vertex_shader, fragment_shader, vertex_shader_fsq, fragment_shader_fsq,bbox_shader_str

import time
import metrics

from plane import fit_plane, project_point_on_plane
from ctypes import c_uint32, cast, POINTER


import sys 

from  detector import apply_yolo

import pandas as pd
import os
from collections import Counter

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

def create_buffers(verts,wed_tcoord,inds,shader0):
    global color_buffer
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
    position = glGetAttribLocation(shader0.program, 'aPosition')
    glEnableVertexAttribArray(position)
    
    # Describe the position data layout in the buffer
    glVertexAttribPointer(position, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,vert_pos.nbytes, vert_pos, GL_STATIC_DRAW)
    
    # Generate buffers to hold our texcoord
    tcoord_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tcoord_buffer)
    
    # Get the position of the 'texcoord' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aTexCoord')
    glEnableVertexAttribArray(position)
    
    # Describe the texcoord data layout in the buffer
    glVertexAttribPointer(position, 2, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    
    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER,tcoords.nbytes, tcoords, GL_STATIC_DRAW)

    # Create an array of n*3 elements as described
    n = len(vert_pos)
    triangle_ids = np.repeat(np.arange(n), 3).astype(np.float32).reshape(-1, 3).flatten()

    # Generate buffers to hold our triangle ids
    triangle_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, triangle_buffer)

    # Get the position of the 'aIdTriangle' in parameter of our shader and bind it.
    position = glGetAttribLocation(shader0.program, 'aIdTriangle')
    glEnableVertexAttribArray(position)

    # Describe the triangle id data layout in the buffer
    glVertexAttribPointer(position, 1, GL_FLOAT, False, 0, ctypes.c_void_p(0))

    # Send the data over to the buffer
    glBufferData(GL_ARRAY_BUFFER, triangle_ids.nbytes, triangle_ids, GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    glBindVertexArray( 0 )
    
    # Unbind other stuff
    glDisableVertexAttribArray(position)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    return vertex_array_object

import matplotlib.pyplot as plt


def display_image(onfluo):
        global id_node
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
        global show_mask

        if(onfluo):
            sensor = sensor_FLUO
        else:
            sensor = sensors[cameras[maskout.all_masks.nodes[id_node].mask.id_camera].sensor_id]
        

        current_unit = glGetIntegerv(GL_ACTIVE_TEXTURE)
        current_texture = glGetIntegerv(GL_TEXTURE_BINDING_2D)
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glClearColor (1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if(onfluo):
            mask = all_masks_fluo[maskout.all_masks.nodes[id_node].mask.id_mask_fluo] #fix index
            update_fluo_textures()
        else:
            mask = maskout.all_masks.nodes[id_node].mask

        glUseProgram(shader_fsq.program)
        glUniform1i(shader_fsq.uni("resolution_width"), sensor.resolution["width"])
        glUniform1i(shader_fsq.uni("resolution_height"), sensor.resolution["height"])
        glUniform1i(shader_fsq.uni("uFluo"), onfluo)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        glUniform1i(shader_fsq.uni("uColorTex"),0)

        if( onfluo): #TODO fix for fluo
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
            glUniform1i(shader_fsq.uni("uColorTex"),0) 

        # Get the currently bound texture on GL_TEXTURE_2D

        glActiveTexture(GL_TEXTURE8)
        current_texture = glGetIntegerv(GL_TEXTURE_BINDING_2D)
        glBindTexture(GL_TEXTURE_2D, mask.id_texture)
        glUniform1i(shader_fsq.uni("uMask"),8)
        glUniform2i(shader_fsq.uni("uOff"),mask.X,mask.Y)
        glUniform2i(shader_fsq.uni("uSize"),mask.w,mask.h)

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

        #zoom stuff
        if not show_mask:
            glUniform1f(shader_fsq.uni("uSca"), 0.0)
        else:
            glUniform1f(shader_fsq.uni("uSca"), curr_zoom )
        glUniform2f(shader_fsq.uni("uTra"), tra.x, tra.y)  


        glActiveTexture(current_unit)
        glBindTexture(GL_TEXTURE_2D, current_texture)
        glUseProgram(0)

def camera_matrix_RGB(id_camera):
    mat4_np = np.eye(4)
    mat4_np[:3, :3] = chunk_rot.reshape(3, 3)
    chunk_rot_matrix =  glm.transpose(glm.mat4(*mat4_np.flatten()))
    chunk_tra_matrix =  glm.translate(glm.mat4(1.0), glm.vec3(*chunk_transl))
    chunk_sca_matrix =  glm.scale(glm.mat4(1.0),  glm.vec3(chunk_scal))
    chunk_matrix = chunk_tra_matrix* chunk_sca_matrix* chunk_rot_matrix
    camera_frame = chunk_matrix * (glm.transpose(glm.mat4(*cameras[id_camera].transform)))
    camera_matrix = glm.inverse(camera_frame)
    return camera_matrix,camera_frame

def camera_matrix_FLUO(id_camera):
  #  mat4_np = np.eye(4)
  #  mat4_np[:3, :3] = chunk_rot_FLUO.reshape(3, 3)
  #  chunk_rot_matrix =  glm.transpose(glm.mat4(*mat4_np.flatten()))
  #  chunk_tra_matrix =  glm.translate(glm.mat4(1.0), glm.vec3(*chunk_transl_FLUO))
  #  chunk_sca_matrix =  glm.scale(glm.mat4(1.0),  glm.vec3(chunk_scal_FLUO))
  #  chunk_matrix = chunk_tra_matrix* chunk_sca_matrix* chunk_rot_matrix
  #  camera_frame = chunk_matrix * (glm.transpose(glm.mat4(*cameras_FLUO[id_camera].transform)))
#    camera_frame = glm.transpose(glm.mat4(*transf_FR.flatten()))* camera_frame #apply the alignment transformation

    
    camera_frame =   transf_FR*chunk_matrix_FLUO(id_camera)* glm.transpose(glm.mat4(*cameras_FLUO[id_camera].transform)) #apply the alignment transformation
    camera_matrix = glm.inverse(camera_frame)
    return camera_matrix,camera_frame

def chunk_matrix_FLUO(id_camera):
    mat4_np = np.eye(4)
    mat4_np[:3, :3] = chunk_rot_FLUO.reshape(3, 3)
    chunk_rot_matrix =  glm.transpose(glm.mat4(*mat4_np.flatten()))
    chunk_tra_matrix =  glm.translate(glm.mat4(1.0), glm.vec3(*chunk_transl_FLUO))
    chunk_sca_matrix =  glm.scale(glm.mat4(1.0),  glm.vec3(chunk_scal_FLUO))
    chunk_matrix = chunk_tra_matrix* chunk_sca_matrix* chunk_rot_matrix
    print(f"chunk_matrix_FLUO for camera {id_camera}:\n{chunk_matrix}")
    print(f"determinant: {glm.determinant(chunk_matrix)}")
    return chunk_matrix

def display(shader0, r,tb,detect,get_uvmap):
    global polyps
    global show_image
    global user_matrix
    global projection_matrix
    global view_matrix
    global user_camera
    global id_camera
    global cameras
    global texture_IMG_id
    global vao_fsq
    global project_image
    global W
    global H
    global id_comp  
    global id_node 
    global show_all_masks 
    global show_all_comps
    global chunk_rot
    global chunk_transl
    global chunk_scal

    global id_camera_fluo

    if(detect):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_uv.id_fbo)
        glDrawBuffers([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2,GL_COLOR_ATTACHMENT3])
        glViewport(0,0,fbo_uv.w,fbo_uv.h)
    else:
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glViewport(0,0,W,H)

    glClearColor (1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader0.program)

    camera_matrix,camera_frame = camera_matrix_RGB(id_camera)
    cameras[id_camera].frame = camera_frame
   
    if(user_camera and not detect):
        # a view of the scene
        view_matrix = user_matrix
        projection_matrix = glm.perspective(glm.radians(45), 1.5,0.1,10)
        glUniformMatrix4fv(shader0.uni("uProj"),1,GL_FALSE, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(shader0.uni("uTrack"), 1, GL_FALSE, glm.value_ptr(tb_matrix := tb.matrix()))
    else:
        # the view from the current id_camera
        view_matrix = camera_matrix
        maskout.current_camera_matrix = camera_matrix

    glUniformMatrix4fv(shader0.uni("uView"),1,GL_FALSE,  glm.value_ptr(view_matrix))
    glUniform1i(shader0.uni("uMode"),user_camera)
    glUniform1i(shader0.uni("uModeProj"),project_image)

    set_sensor(shader0,sensors[cameras[id_camera].sensor_id])

    glActiveTexture(GL_TEXTURE0)
    if(project_image):
        # texture the geometry with the current id_camera image
        glBindTexture(GL_TEXTURE_2D, texture_IMG_id)
        glUniformMatrix4fv(shader0.uni("uViewCam"),1,GL_FALSE,  glm.value_ptr(camera_matrix))
    else:
        # use the texture of the mesh
        glBindTexture(GL_TEXTURE_2D, r.texture_id)

    #print(f"mat: {projection_matrix*view_matrix}, \n P(0)={projection_matrix*view_matrix*tb.matrix()*glm.vec4(0,0,0,1)}")
    #draw the geometry

    glBindVertexArray( r.vao )
    glDrawArrays(GL_TRIANGLES, 0, r.n_faces*3  )
    glBindVertexArray( 0 )

    if not detect: 
        #glDisable(GL_DEPTH_TEST)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(1.0, 1.0)
        for idx, pol in enumerate(polyps):
            if show_all_comps or idx == id_comp:
                glPointSize(5)
                glColor3f(0.0, 0.0, 1.0)

                #mat = glm.mul(tb.matrix(), glm.translate(glm.mat4(1.0), glm.vec3(*pol.centroid_3D)))
                mat =  tb.matrix()* glm.translate(glm.mat4(1.0), glm.vec3(*pol.centroid_3D))

                glUniformMatrix4fv(shader0.uni("uTrack"),1,GL_FALSE, glm.value_ptr(mat))
                gluSphere(quadric,0.0005,8,8)
                glUniformMatrix4fv(shader0.uni("uTrack"),1,GL_FALSE, glm.value_ptr(tb_matrix := tb.matrix()))

            # glBegin(GL_POINTS)
                #glVertex3fv(pol.centroid_3D)
                
            # glColor3f(1.0,0.0,0.0)
                #for sample in pol.samples:
                #    glVertex3fv(sample) 
            # glEnd()

                #glBegin(GL_LINES)
                #glColor3f(1.0, 1.0, 1.0)
                #glVertex3fv(pol.centroid_3D)
                #glVertex3fv(pol.tip_0)
                #glColor3f(1.0, 0.0, 0.0)
                #glVertex3fv(pol.centroid_3D)
                #glVertex3fv(pol.tip_1)
                #glEnd()

        glDisable(GL_POLYGON_OFFSET_LINE)
        #glEnable(GL_DEPTH_TEST)

        if False:
            for idx, pol in enumerate(polyps):
                if show_all_comps or idx == id_comp:
                    glBegin(GL_LINES)
                    glColor3f(0.0, 1.0, 0.0)
                    glVertex3fv(pol.centroid_3D)
                    glVertex3fv(pol.normal_tip)
                    #glVertex3fv(pol.centroid_3D)
                    #glVertex3fv(pol.centroid_3D + (pol.centroid_3D-pol.normal_tip))
                    glEnd()


    #  draw the camera frames
    if(not detect  and  user_camera):
        for i in range(0,len(cameras)):
             if(i == id_camera):
               # camera_frame = chunk_matrix * ((glm.transpose(glm.mat4(*cameras[i].transform))))
               # _,camera_frame = camera_matrix_RGB(i)
                track_mul_frame = tb.matrix()*camera_frame

                glUniformMatrix4fv(shader0.uni("uTrack"),1,GL_FALSE, glm.value_ptr(track_mul_frame))

                glBindVertexArray(vao_frame )
                glDrawArrays(GL_LINES, 0, 6)
                glBindVertexArray( 0 )

        if show_fluo_camera:
            for i in range(0,len(cameras_FLUO)):
                if(i == id_camera_fluo):
                    if i%4 == 2:#only show every forth camera (the green ones)
                        #camera_frame = transf_FR*  chunk_matrix_FLUO(i) * ((glm.transpose(glm.mat4(*cameras_FLUO[i].transform))))
                        _,camera_frame = camera_matrix_FLUO(i)
                        track_mul_frame = tb.matrix()*camera_frame

                        glUniformMatrix4fv(shader0.uni("uTrack"),1,GL_FALSE, glm.value_ptr(track_mul_frame))

                        

                        gluSphere(quadric,0.05,8,8)


                        glBindVertexArray(vao_frame )
                        glDrawArrays(GL_LINES, 0, 6)
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

    
    if False and detect and get_uvmap :
        glBindFramebuffer(GL_FRAMEBUFFER,0)

        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex1)
        buf =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf)
        maskout.uv_map =np.flipud(np.frombuffer(buf, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        uv_map_uint8 = (maskout.uv_map * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_color_{id_camera}.png")   


        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex2)
        buf = np.empty((fbo_uv.h, fbo_uv.w, 3), dtype=np.float32)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf)
        maskout.triangle_map =np.flipud(buf)
        uv_map_uint8 = (maskout.triangle_map * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_idtriangles_{id_camera}.png")    
 

    if False and get_uvmap:
        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_depth)

        colatt1 =   bytearray(fbo_uv.h* fbo_uv.w* 4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, colatt1)
        colatt1 =np.flipud(np.frombuffer(colatt1, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w,1)))
        uv_map_uint8 = (colatt1 * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8.squeeze(), 'L')
        image.save(f"output_depth_{id_camera}.png")     
        

    if  False and maskout.DBG_writeout :
            
        # save the uvmap
        uv_map_uint8 = (maskout.uv_map * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_uvmap_{id_camera}.png")        
        
        #read the colorattachment
        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex)

        colatt1 =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, colatt1)
        colatt1 =np.flipud(np.frombuffer(colatt1, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        uv_map_uint8 = (colatt1 * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_color_{id_camera}.png")     
        
        glBindTexture(GL_TEXTURE_2D,fbo_uv.id_tex2)
        idtriangles =   bytearray(fbo_uv.h* fbo_uv.w* 3*4)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, idtriangles)
        idtriangles =np.flipud(np.frombuffer(idtriangles, dtype=np.float32).reshape((fbo_uv.h, fbo_uv.w, 3)))
        uv_map_uint8 = (idtriangles * 255).clip(0, 255).astype(np.uint8)
        # save the color
        image = Image.fromarray(uv_map_uint8 , 'RGB')
        image.save(f"output_idtriangles_{id_camera}.png")     


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

def load_camera_image( id):
    global id_loaded
    global texture_IMG_id
    filename =   imgs_path +"/"+ cameras[id].label+".JPG" 
    print(f"loading {cameras[id].label}.JPG")
    glDeleteTextures(1, [texture_IMG_id])
    texture_IMG_id,_,__ = texture.load_texture(filename)
    id_loaded = id

def load_mesh(filename):
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

def refresh_domain():
    glActiveTexture(GL_TEXTURE3)
    glBindTexture(GL_TEXTURE_2D, rend.mask_id)
    datatoload = np.flipud(maskout.domain_mask).astype(np.uint8) 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_w, texture_h, 0,  GL_RGB,  GL_UNSIGNED_BYTE, datatoload)

    curr_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)
    curr_vbo = glGetIntegerv(GL_ARRAY_BUFFER_BINDING)
    glBindVertexArray(rend.vao)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    
    glBufferData(GL_ARRAY_BUFFER,maskout.tri_color.nbytes, maskout.tri_color, GL_STATIC_DRAW)
    glBindVertexArray(curr_vao)
    glBindBuffer(GL_ARRAY_BUFFER, curr_vbo)



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
   



def main():
    glm.silence(4)
    global W
    global H
    W = 1200
    H = 800
    global masks_filenames
    global tb

    global sensors
    global vertices 
    global chunk_rot
    global chunk_transl
    global chunk_scal
    global global_transf
    global vao_frame
    global shader_fsq
    global texture_IMG_id
    global show_image
    global vao_fsq
    global id_loaded
    global project_image
    global detect
    global masks_path
    global rend
    global shader0
    global user_matrix
    global id_mask_to_load
    global polyps
    global app_path
    global main_path
    global imgs_path
    global imgs_path_FLUO
    global metashape_file
    global metashape_file_FLUO
    global transf_FLUO_RGB
    global show_all_masks
    global show_all_comps
    global id_comp 
    global id_node 
    global range_threshold
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
    global quadric
    global cov_thr
    global show_fluo_camera
    global texture_w
    global texture_h
    global all_values_fluo
    global pctl
    global fluo_thr #ignore values below this threshold

    all_values_fluo = []
    pctl = [50,75,90,95]
    fluo_thr = 0.0
    

    show_mask = False
    is_translating = False

    np.random.seed(42)  # For reproducibility

    global FLUO 
    FLUO  = False

    # with open("last.txt", "r") as f:
    #     content = f.read()
    #     print("Raw content:", repr(content))
    #     transf_FLUO_RGB = None
    #     lines = content.splitlines()
    #     print("lines:", lines)
    #     if len(lines) >= 5:
    #         main_path = lines[0]
    #         imgs_path = lines[1]
    #         masks_path = lines[2]
    #         mesh_name = lines[3]
    #         metashape_file = lines[4]
    #         if len(lines) == 8:
    #             imgs_path_FLUO = lines[5]
    #             metashape_file_FLUO = lines[6]
    #             transf_FLUO_RGB = lines[7]
    #             if transf_FLUO_RGB != '':
    #                 FLUO = True
    #     else:
    #         print("last.txt does not contain enough lines.")

    transf_FLUO_RGB = None
    FLUO = False

    try:
        with open("last.json", "r") as f:
            data = json.load(f)
            print("Raw content:", data)

            main_path = data.get("main_path")
            imgs_path = data.get("imgs_path")
            masks_path = data.get("masks_path")
            mesh_name = data.get("mesh_name")
            metashape_file = data.get("metashape_name")
            global_transf_name = data.get("global_transf")

            global_transf =  glm.mat4(1.0)
            if global_transf_name not in (None, ""):
                global_transf = glm.mat4(*np.loadtxt(global_transf_name, delimiter=' ').T.flatten())



            # Optional FLUO fields
            imgs_path_FLUO = data.get("imgs_path_FLUO")
            metashape_file_FLUO = data.get("metashape_name_FLUO")
            transf_FLUO_RGB = data.get("transf_FLUO_RGB")

            if transf_FLUO_RGB not in (None, ""):
                FLUO = True

    except FileNotFoundError:
        print("last.json not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format in last.json.")
    

    if len(sys.argv) > 1:
        main_path = sys.argv[1]
        imgs_path = sys.argv[2]
        masks_path = sys.argv[3]
        mesh_name = sys.argv[4]
        metashape_file = sys.argv[5]
        if len(sys.argv) == 8:
            imgs_path_FLUO = sys.argv[6]
            metashape_file_FLUO = sys.argv[7]
            transf_FLUO_RGB = sys.argv[8]
            if transf_FLUO_RGB != '':
                FLUO = True

    print(f"params: {sys.argv}")

    polyps = []
    id_mask_to_load = 0
    id_loaded = -1
    show_image = False
    project_image = False
    show_fluo_camera = False

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

    tb = trackball.Trackball()
    tb.reset()
    glClearColor(1, 1,1, 0.0)
    glEnable(GL_DEPTH_TEST)
    
    app_path = os.path.dirname(os.path.abspath(__file__))
    print(f"App path: {app_path}")


    os.chdir(main_path)
    vertices, faces, wed_tcoords, bmin,bmax,texture_id,texture_w,texture_h  = load_mesh(mesh_name) 

 
    global cameras_FLUO 
    global chunk_rot_FLUO 
    global chunk_transl_FLUO 
    global chunk_scal_FLUO
    global cameras 
    global sensor_FLUO



        
    sensors = metashape_loader.load_sensors_from_xml(metashape_file)




    chunk_rot = [1,0,0,0,1,0,0,0,1]
    chunk_transl = [0,0,0]
    chunk_scal = 1

    cameras,chunk_rot,chunk_transl,chunk_scal = metashape_loader.load_cameras_from_xml(metashape_file)

    if chunk_rot is None:
        chunk_rot = [1,0,0,0,1,0,0,0,1]
    if chunk_transl is None:
        chunk_transl = [0,0,0]
    if chunk_scal is None:
        chunk_scal = 1
        




    chunk_rot = np.array(chunk_rot)
    chunk_transl = np.array(chunk_transl)

    
    rend = renderable(vao=None,n_verts=len(vertices),n_faces=len(faces),texture_id=texture_id,mask_id=texture.create_texture(texture_w,texture_h))

    global shader0
    global shader_fluo

    shader0     = shader(vertex_shader, fragment_shader)
    shader_fsq  = shader(vertex_shader_fsq, fragment_shader_fsq)
   

    check_gl_errors()










    #faces = np.array(faces, dtype=np.uint32).flatten()
    print(f"vertices: {len(vertices) }")
    print(f"faces: {len(faces)}")

    #print(faces)
    vertex_array_object = create_buffers(vertices,wed_tcoords,faces,shader0)
    rend.vao = vertex_array_object
    vao_frame = create_buffers_frame()
    vao_fsq = create_buffers_fsq()

    global viewport
    clock = pygame.time.Clock()
    viewport =[0,0,W,H]
    center = (bmin+bmax)/2.0
    eye = center + glm.vec3(2,0,0)
    user_matrix = glm.lookAt(glm.vec3(eye),glm.vec3(center), glm.vec3(0,0,1)) # TODO: UP PARAMETRICO !
    projection_matrix = glm.perspective(glm.radians(45),1.5,0.1,10) # TODO: NEAR E FAR PARAMETRICI !!
    tb.set_center_radius(center, 1.0)
    


    global id_camera
    global user_camera
    id_camera   = 0

    show_mask_fluo = False
    user_camera = 1
    detect = False
    #go_process_masks = False
    go_process_all_masks = False
    i_toload = 0
    id_comp = 0
    id_node = 0
    texture_IMG_id = 0
    th = 10
    prevtime = 0
  #  show_metrics = False
    show_all_masks = True
    show_all_comps  = True
    range_threshold = 100
    mask_zoom = 1.0
    mask_xpos = W/2
    mask_ypos = H/2
    curr_zoom = 1.0
    curr_center = glm.vec2(0,0)
    tra_xstart = mask_xpos
    tra_ystart = mask_ypos
    curr_tra = glm.vec2(0,0)
    cov_thr = 0.6



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
                if show_image and is_translating:
                    mask_xpos = mouseX
                    mask_ypos = mouseY
                else:    
                    tb.mouse_move(projection_matrix, user_matrix, mouseX, mouseY)

            if event.type == pygame.MOUSEWHEEL:
                xoffset, yoffset = event.x, event.y
                if show_image:
                     mask_zoom = 1.1 if yoffset > 0 else 0.97
                     if yoffset > 0 :
                        mask_xpos = mouseX 
                        mask_ypos = mouseY
                else:
                    tb.mouse_scroll(xoffset, yoffset)
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
                                tb.reset_center(cp)               
                        else:
                            tb.mouse_press(projection_matrix, user_matrix, mouseX, mouseY)
            if event.type == pygame.MOUSEBUTTONUP:
                mouseX, mouseY  = event.pos
                if event.button == 1:  # Left mouse button
                    if show_image:
                        is_translating = False
                    else:
                        tb.mouse_release()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    user_camera = 1 - user_camera

        imgui.new_frame()

        if imgui.begin_main_menu_bar().opened:

            # first menu dropdown
            if imgui.begin_menu('Actions', True).opened:

                imgui.end_menu()
            

                  
        imgui.end_main_menu_bar()
            

    
        check_gl_errors()
        if show_image:
            display_image(show_mask_fluo)
        else:
            display(shader0, rend,tb, False,False) 

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
