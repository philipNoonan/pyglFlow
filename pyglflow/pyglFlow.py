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



def do_gradFilter(gradShader, imageTex, outImage, level, width, height):
    glUseProgram(gradShader)
    glBindImageTexture(0, imageTex, level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8UI)
    glBindImageTexture(1, outImage, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F)

    lesserID = glGetUniformLocation(gradShader, "lesser")
    upperID = glGetUniformLocation(gradShader, "upper")
    normID = glGetUniformLocation(gradShader, "normVal")
    frameID = glGetUniformLocation(gradShader, "frameCounter")

    lesser = 3
    upper = 10
    norm = 1.0 / (2.0 * 10 + 4.0 * 3)
    glUniform1f(lesserID, lesser)
    glUniform1f(upperID, upper)
    glUniform1f(normID, norm)

    #glUniform1i(frameID, frame)
    xx = int(((int(width) >> level)/32.0)+0.5)
    yy = int(((int(height) >> level)/32.0)+0.5)

    glDispatchCompute(xx, yy, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def do_inverseSearch(inverseSearchShader, textureList, level, width, height):

    invDenseWidth =  1.0 / float(int(width) >> level)
    invDenseHeight = 1.0 / float(int(height) >> level)

    invPrevDenseWidth =  1.0 / float(int(width) >> (level + 1))
    invPrevDenseHeight = 1.0 / float(int(height) >> (level + 1))

    glUseProgram(inverseSearchShader)

    glUniform1i(glGetUniformLocation(inverseSearchShader, "lastColorMap"), 0)
    glUniform1i(glGetUniformLocation(inverseSearchShader, "nextColorMap"), 1)

    glUniform1i(glGetUniformLocation(inverseSearchShader, "level"), level)
    glUniform2f(glGetUniformLocation(inverseSearchShader, "invImageSize"), invDenseWidth, invDenseHeight)
    glUniform2f(glGetUniformLocation(inverseSearchShader, "invPreviousImageSize"), invPrevDenseWidth, invPrevDenseHeight)


    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureList[0]) # last col
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textureList[1]) # next col

    glBindImageTexture(0, textureList[2], level, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F) # last grad
    glBindImageTexture(1, textureList[5], int(level + 1), GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) # flow
    glBindImageTexture(2, textureList[6], level, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F) # sparse flow


    glBindImageTexture(3, textureList[5], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) # flow to wipe next flow (densified flow)
    glBindImageTexture(4, textureList[4], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) # last flow

    sparseWidth = (int(width / 4) >> level)
    sparseHeight = (int(height / 4) >> level)

    compWidth = int((sparseWidth/32.0)+0.5)
    compHeight = int((sparseHeight/32.0)+0.5)


    glDispatchCompute(int((sparseWidth/32.0)+0.5), int((sparseHeight/32.0)+0.5), 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)


def do_densify(densifyShader, framebuffers, textureList, level, width, height):
    glUseProgram(densifyShader)

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[level])


   # glDisable(GL_DEPTH_TEST)

    glEnable(GL_BLEND)
    glBlendFunc(GL_ONE, GL_ONE)

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

    glViewport(0,0, int(width + 0.5) >> level, int(height + 0.5) >> level)

    invDenseWidth =  1.0 / float(int(width) >> level)
    invDenseHeight = 1.0 / float(int(height) >> level)

    sparseWidth = (int(width / 4) >> level)
    sparseHeight = (int(height / 4) >> level)

    glUniform1i(glGetUniformLocation(densifyShader, "level"), level)
    glUniform2f(glGetUniformLocation(densifyShader, "invDenseTexSize"), invDenseWidth, invDenseHeight)
    glUniform2i(glGetUniformLocation(densifyShader, "sparseTexSize"), int(sparseWidth), int(sparseHeight))

    drawBuffs = [GL_COLOR_ATTACHMENT0]


    glBindImageTexture(0, textureList[6], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) # sparse flow


    glUniform1i(glGetUniformLocation(densifyShader, "lastImage"), 0)
    glUniform1i(glGetUniformLocation(densifyShader, "nextImage"), 1)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureList[0]) # last col
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textureList[1]) # next col

    glDrawBuffers(1, drawBuffs)

    numberOfPatches = sparseWidth * sparseHeight

    glDrawArrays(GL_POINTS, 0, int(numberOfPatches))

    glBindFramebuffer(GL_FRAMEBUFFER, 0)


   # glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE)




def read_texture_memory(imageTex, width, height):

    newImages = np.empty([width*height*4], dtype=np.float)
    #glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, imageTex)
    newImages = glGetTexImageub(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)

    newImages.reshape((width, height, 4))
    print(newImages)

def createTexture(texture, target, internalFormat, levels, width, height, depth, minFilter, magFilter):

    if texture == -1:
        texName = glGenTextures(1)
    else:
        glDeleteTextures(int(texture))
        texName = texture
        texName = glGenTextures(1)

    glBindTexture(target, texName)
    #texture wrapping params
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    #texture filtering params
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter)
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter)
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

def generateTextures(textureList, numImages, width, height):

    
    maxLevels = 4 # FIXME
    numLevels = 4 # FIXME TOO
    #lastColor
    textureList[0] = createTexture(textureList[0], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #nextColor
    textureList[1] = createTexture(textureList[1], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #lastGradMap
    textureList[2] = createTexture(textureList[2], GL_TEXTURE_2D, GL_RG32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #nextGradMap
    textureList[3] = createTexture(textureList[3], GL_TEXTURE_2D, GL_RG32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #lastFlowMap
    textureList[4] = createTexture(textureList[4], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #nextFlowMap
    textureList[5] = createTexture(textureList[5], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #sparseFlowMap
    textureList[6] = createTexture(textureList[6], GL_TEXTURE_2D, GL_RGBA32F, maxLevels, int(width / 4), int(height / 4), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #densificationFlowMap
    #textureList[7] = createTexture(textureList[7], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

 
    

	# Allocate the immutable GPU memory storage -more efficient than mutable memory if you are not going to change image size after creation
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, int(width))



    return textureList

def generateDensificationFramebuffer(densificationFlowMap, width, height):
    
    maxLevels = 4

    framebuffers = np.empty(maxLevels, dtype=np.uint32)
    glCreateFramebuffers(maxLevels, framebuffers)

    for i in range(maxLevels):
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[i])
        depthTex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, depthTex)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, int(width) >> i, int(height) >> i, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, None)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, densificationFlowMap, i)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print("framebuffer incomplete!")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


    return framebuffers




def main():

    # NAMED TEXTURES
    lastColor = -1            # 0
    nextColor = -1            # 1
    lastGradMap = -1          # 2
    nextGradMap = -1          # 3
    lastFlowMap = -1          # 4
    nextFlowMap = -1          # 5
    sparseFlowMap = -1        # 6
    densificationFlowMap = -1 # 7



    textureList = [lastColor, nextColor, lastGradMap, nextGradMap, lastFlowMap, nextFlowMap, sparseFlowMap, densificationFlowMap]

    densifiactionFBO = -1

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

    vertex_shader = (Path(__file__).parent / 'shaders/screenQuad.vert').read_text()

    fragment_shader = (Path(__file__).parent / 'shaders/screenQuad.frag').read_text()

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))


    grad_shader = (Path(__file__).parent / 'shaders/gradient.comp').read_text()

    gradShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(grad_shader, GL_COMPUTE_SHADER))

    inverseSearch_shader = (Path(__file__).parent / 'shaders/disSearch.comp').read_text()

    inverseSearchShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(inverseSearch_shader, GL_COMPUTE_SHADER))

    densifyVert_shader = (Path(__file__).parent / 'shaders/disDensification.vert').read_text()
    densifyFrag_shader = (Path(__file__).parent / 'shaders/disDensification.frag').read_text()

    densifyShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(densifyVert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(densifyFrag_shader, GL_FRAGMENT_SHADER))

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
    listy = list(Path('./data/').glob('./*'))
    for x in listy:
        fileList.append(str(x))

    currentCamera = 0
    cameraList = ['0', '1', '2', '3', '4']
    resetVideoSource = True
    sourceAvailable = False

    global cap

    filemode = 0 # 1 : webcam, 2 : video file

    numberOfImages = 1000

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

                textureList = generateTextures(textureList, numberOfImages, width, height)
                densifiactionFBO = generateDensificationFramebuffer(textureList[5], width, height)

                
                resetVideoSource = False

        else:
            ret, frame = cap.read()

            if ret:

                glBindFramebuffer(GL_FRAMEBUFFER, 0)

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, textureList[0])

                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


                img_data = np.array(frame.data, np.uint8)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_BGR, GL_UNSIGNED_BYTE, img_data)
                glGenerateMipmap(GL_TEXTURE_2D)


                if (doFilterEnabled):
                    for lvl in range(3, -1, -1):
                        do_gradFilter(gradShader, textureList[0], textureList[2], lvl, width, height)
                        do_inverseSearch(inverseSearchShader, textureList, lvl, width, height)
                        do_densify(densifyShader, densifiactionFBO, textureList, lvl, width, height)

                w, h = glfw.get_framebuffer_size(window)

                # set the active drawing viewport within the current GLFW window (i.e. we are spliiting it up in 3 cols)
                xpos = 0
                ypos = 0
                xwidth = float(w) / 3.0
                glViewport(int(xpos), int(ypos), int(xwidth),h)
                glClear(GL_COLOR_BUFFER_BIT)

                glUseProgram(shader)


                glUniform1i(renderType_loc, 0)
                glUniform1i(sliderR_loc, sliderRValue)

                #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
                # DRAW THE FIRST WINDOW (live feed)
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)



                # set second draw call's drawing location (we've shifted accros by width / 4)
                xpos = w / 3.0                
                glViewport(int(xpos), int(ypos), int(xwidth),h)
                glUniform1i(renderType_loc, 1)

                # we want to now render from the processed texture, whose memory has been populated by the edgeFilter compute shader
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, textureList[2])
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                glUniform1i(sliderR_loc, sliderRValue)


                # set third draw call's drawing location (we've shifted accros by 2 * width / 3)
                xpos = 2.0 * float(w) / 3.0
                glViewport(int(xpos), int(ypos), int(xwidth),h)
                glUniform1i(renderType_loc, 2)

                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, textureList[6])

                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


                # set third draw call's drawing location (we've shifted accros by 2 * width / 3)
                # xpos = 3.0 * float(w) / 3.0
                # glViewport(int(xpos), int(ypos), int(xwidth),h)
                # glUniform1i(renderType_loc, 2)

                # glActiveTexture(GL_TEXTURE1)
                # glBindTexture(GL_TEXTURE_2D, textureList[5])

                # glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


                # swap frame handles
                textureList[0], textureList[1] = textureList[1], textureList[0]
                textureList[2], textureList[3] = textureList[3], textureList[2]
                textureList[4], textureList[5] = textureList[5], textureList[4]


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


        changedR, sliderRValue = imgui.slider_int("sliceR", sliderRValue, min_value=0, max_value=5)
        changedG, sliderGValue = imgui.slider_int("sliceG", sliderGValue, min_value=0, max_value=numberOfImages)
        changedB, sliderBValue = imgui.slider_int("sliceB", sliderBValue, min_value=0, max_value=numberOfImages)
        _, doFilterEnabled = imgui.checkbox("run filter", doFilterEnabled)




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
