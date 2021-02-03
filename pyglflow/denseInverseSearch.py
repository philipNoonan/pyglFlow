import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np

def do_gradFilter(gradShader, textureList, level, width, height):
    glUseProgram(gradShader)

    glUniform1i(glGetUniformLocation(gradShader, "colorTex"), 0)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureList[0]) # last col

    lvlID = glGetUniformLocation(gradShader, "level")
    glUniform1i(lvlID, level)

    glBindImageTexture(0, textureList[2], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F)

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

    lcID = glGetUniformLocation(inverseSearchShader, "lastColorMap")
    glUniform1i(lcID, 0)
    ncID = glGetUniformLocation(inverseSearchShader, "nextColorMap")
    glUniform1i(ncID, 1)

    lvlID = glGetUniformLocation(inverseSearchShader, "level")
    glUniform1i(lvlID, level)

    iisID = glGetUniformLocation(inverseSearchShader, "invImageSize")
    glUniform2f(iisID, invDenseWidth, invDenseHeight)

    #ipisID = glGetUniformLocation(inverseSearchShader, "invPreviousImageSize")
    #glUniform2f(ipisID, invPrevDenseWidth, invPrevDenseHeight)


    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureList[0]) # last col
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textureList[1]) # next col

    glBindImageTexture(0, textureList[2], level, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F) # last grad
    glBindImageTexture(1, textureList[4], int(level + 1), GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) # flow to read from
    glBindImageTexture(2, textureList[6], level, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F) # sparse flow


    glBindImageTexture(3, textureList[4], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) # flow to wipe next flow (densified flow)
    glBindImageTexture(4, textureList[5], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) # last flow

    sparseWidth = (int(width / 4) >> level)
    sparseHeight = (int(height / 4) >> level)

    compWidth = int((sparseWidth/32.0)+0.5)
    compHeight = int((sparseHeight/32.0)+0.5)


    glDispatchCompute(int((sparseWidth/32.0)+0.5), int((sparseHeight/32.0)+0.5), 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)


def do_densify(densifyShader, framebuffers, textureList, level, width, height):
    glUseProgram(densifyShader)

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[level])


    #glDisable(GL_DEPTH_TEST)

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


    #glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE)

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