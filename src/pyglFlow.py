import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from glob import glob
import cv2
import re
import os
import imgui
from imgui.integrations.glfw import GlfwRenderer
from pathlib import Path



def do_edgeFilter(edgeShader, imageTex, outImage, frame, width, height):
    glUseProgram(edgeShader)
    glBindImageTexture(0, imageTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8UI)    
    glBindImageTexture(1, outImage, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)    

    lesserID = glGetUniformLocation(edgeShader, "lesser")
    upperID = glGetUniformLocation(edgeShader, "upper")
    normID = glGetUniformLocation(edgeShader, "normVal")
    frameID = glGetUniformLocation(edgeShader, "frameCounter")

    lesser = 3
    upper = 10
    norm = 1.0 / (2.0 * 10 + 4.0 * 3)
    glUniform1f(lesserID, lesser)
    glUniform1f(upperID, upper)
    glUniform1f(normID, norm)

    glUniform1i(frameID, frame)

    glDispatchCompute(int((width/32.0)+0.5), int((height/32.0)+0.5), 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def read_texture_memory(imageTex, width, height):

    newImages = np.empty([width*height*4], dtype=np.float)
    #glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, imageTex)
    newImages = glGetTexImageub(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    
    newImages.reshape((width, height, 4))
    print(newImages)	

def createTexture(texture, target, internalFormat, levels, width, height, depth, minFilter, magFilter):
    print ('name : ', texture)

    if texture == -1:
        texName = glGenTextures(1)
        print ('here : ', texName)
    else:
        glDeleteTextures(int(texture))
        print ('errr : ')

        texName = texture
        texName = glGenTextures(1)
        print ('post : ', texName)

    glBindTexture(target, texName)
    #texture wrapping params
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    #texture filtering params
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter)
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter)
    print(texName, target, internalFormat, levels, width, height, depth, minFilter, magFilter)
    if target == GL_TEXTURE_1D:
        glTexStorage1D(target, levels, internalFormat, width)
    elif target == GL_TEXTURE_2D:
        glTexStorage2D(target, levels, internalFormat, width, height)
    elif target == GL_TEXTURE_3D or depth > 1:
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)	
        glTexStorage3D(target, levels, internalFormat, width, height, depth)

    return texName	

def reset():
    try: 
        cap
    except NameError:
        print('')        
    else:
        cap.release()
    
def openVideo(filename):
    cap = cv2.VideoCapture(filename)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    return cap, width, height

def openCamera(camera):
    cap = cv2.VideoCapture(int(camera))
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    return cap, width, height

def generateTextures(hyperspectralDataTexture, processedTexture, numImages, width, height):

    hyperspectralDataTexture = createTexture(hyperspectralDataTexture, GL_TEXTURE_2D_ARRAY, GL_R8, 1, int(width), int(height), numImages, GL_LINEAR, GL_LINEAR)
    processedTexture = createTexture(processedTexture, GL_TEXTURE_2D, GL_RGBA32F, 1, int(width), int(height), 1, GL_LINEAR, GL_LINEAR)


	# Allocate the immutable GPU memory storage -more efficient than mutable memory if you are not going to change image size after creation
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);	
    glPixelStorei(GL_UNPACK_ROW_LENGTH, int(width))



    return 	hyperspectralDataTexture, processedTexture

def main():

    # NAMED TEXTURES    
    hyperspectralDataTexture = -1
    processedTexture = -1

    # initialize glfw
    if not glfw.init():
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    #creating the window
    window = glfw.create_window(1600, 900, "PyGLFlow", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
	
    imgui.create_context()
    impl = GlfwRenderer(window)
    #           positions        colors          texture coords
    quad = [   -1.0, -1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 0.0,
                1.0, -1.0, 0.0,  0.0, 1.0, 0.0,  1.0, 0.0,
                1.0,  1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 1.0,
               -1.0,  1.0, 0.0,  1.0, 1.0, 1.0,  0.0, 1.0]

    quad = np.array(quad, dtype = np.float32)

    indices = [0, 1, 2,
               2, 3, 0]

    indices = np.array(indices, dtype= np.uint32)

    vertex_shader = Path('../shaders/screenQuad.vert').read_text()

    fragment_shader = Path('../shaders/screenQuad.frag').read_text()

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))


    edge_shader = Path('../shaders/gradient.comp').read_text()

    edgeShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(edge_shader, GL_COMPUTE_SHADER))


    # set up VAO and VBO for full screen quad drawing calls
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
	
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 128, quad, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 24, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    #find the binding locations in the shaders for some uniforms
    sliderR_loc = glGetUniformLocation(shader, "sliderR")
    sliderG_loc = glGetUniformLocation(shader, "sliderG")
    sliderB_loc = glGetUniformLocation(shader, "sliderB")
    renderType_loc = glGetUniformLocation(shader, "renderType")

    
    # make some default background color
    glClearColor(0.2, 0.3, 0.2, 1.0)
    
    # set some values
    sliderRValue = 0		
    sliderGValue = 0		
    sliderBValue = 0

    frameCounter = 0		

    #default to not running any filters
    doFilterEnabled = False
    showFileDialogueOptions = False
    showCameraDialogueOptions = False
    currentFile = 0
    fileList = []
    listy = list(Path('../data/').glob('./*'))
    for x in listy:
        fileList.append(str(x))

    currentCamera = 0
    cameraList = ['0', '1', '2', '3', '4']
    resetVideoSource = True    
    sourceAvailable = False
    
    global cap

    filemode = 0 # 1 : webcam, 2 : video file
  	
    numberOfImages = 1000 
	
    hyperspectralDataTexture = -1
    processedTexture = -1
    width = 0
    height = 0





    while not glfw.window_should_close(window):

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        if resetVideoSource:
            if sourceAvailable:
                reset()
                if filemode == 1:
                    cap, width, height = openCamera(cameraList[currentCamera])
                elif filemode == 2:
                    cap, width, height = openVideo(fileList[currentFile])

                hyperspectralDataTexture, processedTexture = generateTextures(hyperspectralDataTexture, processedTexture, numberOfImages, width, height)
                resetVideoSource = False

        else:
            ret, frame = cap.read()

            if ret:

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D_ARRAY, hyperspectralDataTexture)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


                img_data = np.array(gray.data, np.uint8)		
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, frameCounter, int(width), int(height), 1, GL_RED, GL_UNSIGNED_BYTE, img_data)

                w, h = glfw.get_framebuffer_size(window)

                # set the active drawing viewport within the current GLFW window (i.e. we are spliiting it up in 3 cols)
                glViewport(0,0,int(w/3),h)		
                glClear(GL_COLOR_BUFFER_BIT)

                glUseProgram(shader)

                glUniform1i(sliderR_loc, frameCounter)
                glUniform1i(sliderG_loc, frameCounter)
                glUniform1i(sliderB_loc, frameCounter)
                glUniform1i(renderType_loc, 0)

                #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );	
                # DRAW THE FIRST WINDOW (live feed)	
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                glUniform1i(sliderB_loc, sliderBValue)
                glUniform1i(sliderR_loc, sliderRValue)
                glUniform1i(sliderG_loc, sliderGValue)
                glUniform1i(renderType_loc, 0)

                # set second draw call's drawing location (we've shifted accros by width / 3)
                glViewport(int(w/3),0,int(w/3),h)		
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
                # set third draw call's drawing location (we've shifted accros by 2 * width / 3)
                glViewport(int(2*w/3),0,int(w/3),h)	
                glUniform1i(renderType_loc, 1)
        
                # we want to now render from the processed texture, whose memory has been populated by the edgeFilter compute shader
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, processedTexture)
        
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

            elif ret == False and filemode == 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)



        # GUI TIME
        imgui.begin("Menu", True)

        if imgui.button("Use Camera"):
            filemode = 1
            showCameraDialogueOptions = True

        if imgui.button("Select Video File"):
            filemode = 2
            showFileDialogueOptions = True

        if showCameraDialogueOptions:
            clicked, currentCamera = imgui.combo(
                "cams", currentCamera, cameraList
            )
            if (clicked):
                resetVideoSource = True
                sourceAvailable  = True
                frameCounter = 0

        if showFileDialogueOptions:
            clicked, currentFile = imgui.combo(
                "files", currentFile, fileList
            )
            if (clicked):
                resetVideoSource = True
                sourceAvailable  = True
                frameCounter = 0


        changedR, sliderRValue = imgui.slider_int("sliceR", sliderRValue, min_value=0, max_value=numberOfImages)
        changedG, sliderGValue = imgui.slider_int("sliceG", sliderGValue, min_value=0, max_value=numberOfImages)
        changedB, sliderBValue = imgui.slider_int("sliceB", sliderBValue, min_value=0, max_value=numberOfImages)
        _, doFilterEnabled = imgui.checkbox("run filter", doFilterEnabled)
    
        if (doFilterEnabled):
            do_edgeFilter(edgeShader, hyperspectralDataTexture, processedTexture, frameCounter, width, height)

        _, filemodeCheck = imgui.checkbox("", doFilterEnabled)


        imgui.end()		

        imgui.render()	
	
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)

        frameCounter = frameCounter + 1

        if frameCounter >= numberOfImages:
            frameCounter = 0


    glfw.terminate()
    cap.release()

if __name__ == "__main__":
    main()