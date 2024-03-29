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
import time
import graphics
from array import array
import platform

from collections import deque
import statistics
from scipy.ndimage.filters import uniform_filter1d

# import pycuda.driver as cuda
# import pycuda.autoinit
# import tensorrt as trt

def _add_dll_directory(path: Path):
    from ctypes import c_wchar_p, windll  # type: ignore
    from ctypes.wintypes import DWORD

    AddDllDirectory = windll.kernel32.AddDllDirectory
    AddDllDirectory.restype = DWORD
    AddDllDirectory.argtypes = [c_wchar_p]
    AddDllDirectory(str(path))


def kinect():
    if platform.system() == "Linux":
        return
    env_path = os.getenv("KINECT_LIBS", None)
    if env_path:
        candidate = Path(env_path)
        dll = candidate / "k4a.dll"
        if dll.exists():
            _add_dll_directory(candidate)
            return
    # autodetecting
    program_files = Path("C:\\Program Files\\")
    for dir in sorted(program_files.glob("Azure Kinect SDK v*"), reverse=True):
        candidate = dir / "sdk" / "windows-desktop" / "amd64" / "release" / "bin"
        dll = candidate / "k4a.dll"
        if dll.exists():
            _add_dll_directory(candidate)
            return


def generateBuffers(bufferDict):
    
    bufferDict['sumFlow'] = graphics.createBuffer(bufferDict['sumFlow'], GL_SHADER_STORAGE_BUFFER, 8, 0)

    return bufferDict

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# runtime = trt.Runtime(TRT_LOGGER)
# with open('./models/engine512x320.trt', 'rb') as f:
#     engine_bytes = f.read()
#     engine = runtime.deserialize_cuda_engine(engine_bytes)



#import denseInverseSearch as DIS

if platform.system() == 'Linux':
    import json
    import trt_pose.coco
    import torch2trt
    import torch
    from torch2trt import TRTModule
    import torchvision.transforms as transforms
    import PIL.Image
    from trt_pose.draw_objects import DrawObjects
    from trt_pose.parse_objects import ParseObjects


def divup(a, b):
    if a % b != 0:
        return int(a / b + 1)
    else:
        return int(a / b)

def do_gradFilter(gradShader, textureDict, level, width, height):
    glUseProgram(gradShader)

    glUniform1i(glGetUniformLocation(gradShader, "colorTex"), 0)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['nextColor']) # first col

    lvlID = glGetUniformLocation(gradShader, "level")
    glUniform1i(lvlID, level)


    glBindImageTexture(0, textureDict['nextGradMap'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F)

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
    xx = divup(int(width) >> level, 32)

    yy = divup(int(height) >> level, 32)

    glDispatchCompute(xx, yy, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def do_inverseSearch(inverseSearchShader, textureDict, level, width, height):





    invDenseWidth =  1.0 / float(int(width) >> level)
    invDenseHeight = 1.0 / float(int(height) >> level)

    #invPrevDenseWidth =  1.0 / float(int(width) >> (level + 1))
    #invPrevDenseHeight = 1.0 / float(int(height) >> (level + 1))

    glUseProgram(inverseSearchShader)

    glUniform1i(glGetUniformLocation(inverseSearchShader, "tex_I0"), 0)
    glUniform1i(glGetUniformLocation(inverseSearchShader, "tex_I1"), 1)
    glUniform1i(glGetUniformLocation(inverseSearchShader, "tex_flow_previous"), 2)

    glUniform1i(glGetUniformLocation(inverseSearchShader, "level"), level)
    glUniform1i(glGetUniformLocation(inverseSearchShader, "imageType"), 1)

    #iisID = glGetUniformLocation(inverseSearchShader, "invImageSize")
    #glUniform2f(iisID, invDenseWidth, invDenseHeight)

    #ipisID = glGetUniformLocation(inverseSearchShader, "invPreviousImageSize")
    #glUniform2f(ipisID, invPrevDenseWidth, invPrevDenseHeight)

           


    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['lastColor']) # last col
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textureDict['nextColor']) # col
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, textureDict['nextFlowMap']) # last flow


    glBindImageTexture(0, textureDict['nextGradMap'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F) # grad
    glBindImageTexture(1, textureDict['lastFlowMap'], level, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F) # flow
    glBindImageTexture(2, textureDict['sparseFlowMap'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) # sparse


    sparseWidth = (int(width / 4))
    sparseHeight = (int(height / 4))

    #compWidth = divup(int(width) >> level, 16)
    #compHeight = divup(int(height) >> level, 16)

    compWidth = divup((int(width) >> level), 16)
    compHeight = divup((int(height) >> level), 16)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)


def do_densify(densifyShader, textureDict, level, width, height):
    glUseProgram(densifyShader)

    lcID = glGetUniformLocation(densifyShader, "tex_I0")
    glUniform1i(lcID, 0)
    ncID = glGetUniformLocation(densifyShader, "tex_I1")
    glUniform1i(ncID, 1)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['lastColor']) # last col
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textureDict['nextColor']) # next col

    glBindImageTexture(0, textureDict['sparseFlowMap'], level, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F) # sparse flow
    glBindImageTexture(1, textureDict['nextFlowMap'], level, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F) # dense flow

    glUniform1i(glGetUniformLocation(densifyShader, "imageType"), 1)

    lvlID = glGetUniformLocation(densifyShader, "level")
    glUniform1i(lvlID, level)

    #sparseWidth = (int(width / 4))
    #sparseHeight = (int(height / 4))

    compWidth = divup(int(width) >> level, 32)
    compHeight = divup(int(height) >> level, 32)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

    # glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[level])


    # #glDisable(GL_DEPTH_TEST)

    # glEnable(GL_BLEND)
    # glBlendFunc(GL_ONE, GL_ONE)

    # glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

    # glViewport(0,0, int(width + 0.5) >> level, int(height + 0.5) >> level)

    # invDenseWidth =  1.0 / float(int(width) >> level)
    # invDenseHeight = 1.0 / float(int(height) >> level)

    # sparseWidth = (int(width / 4) >> level)
    # sparseHeight = (int(height / 4) >> level)

    # glUniform1i(glGetUniformLocation(densifyShader, "level"), level)
    # glUniform2f(glGetUniformLocation(densifyShader, "invDenseTexSize"), invDenseWidth, invDenseHeight)
    # glUniform2i(glGetUniformLocation(densifyShader, "sparseTexSize"), int(sparseWidth), int(sparseHeight))

    # drawBuffs = [GL_COLOR_ATTACHMENT0]


    # glBindImageTexture(0, textureList[6], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) # sparse flow


    # glUniform1i(glGetUniformLocation(densifyShader, "lastImage"), 0)
    # glUniform1i(glGetUniformLocation(densifyShader, "nextImage"), 1)

    # glActiveTexture(GL_TEXTURE0)
    # glBindTexture(GL_TEXTURE_2D, textureList[0]) # last col
    # glActiveTexture(GL_TEXTURE1)
    # glBindTexture(GL_TEXTURE_2D, textureList[1]) # next col

    # glDrawBuffers(1, drawBuffs)

    # numberOfPatches = sparseWidth * sparseHeight

    # glDrawArrays(GL_POINTS, 0, int(numberOfPatches))

    # glBindFramebuffer(GL_FRAMEBUFFER, 0)


    # #glEnable(GL_DEPTH_TEST)
    # glDisable(GL_BLEND)
    # glDisable(GL_VERTEX_PROGRAM_POINT_SIZE)

def do_summing(summingShader, textureDict, bufferDict, level, width, height, xpix, ypix):
    glUseProgram(summingShader)

    glBindImageTexture(0, textureDict['nextFlowMap'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) # dense flow
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['sumFlow'])
    glUniform2i(glGetUniformLocation(summingShader, "patchCenter"), int(xpix), int(ypix))

    glDispatchCompute(1, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['sumFlow'])

    tempData = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8)

    sumsFlow = np.frombuffer(tempData, dtype=np.float32)

    #print(sumsFlow)

    return np.float32(sumsFlow[1])

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
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER )
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER )
    #texture filtering params
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter)
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter)
    if target == GL_TEXTURE_1D:
        glTexStorage1D(target, levels, internalFormat, width)
    elif target == GL_TEXTURE_2D:
        glTexStorage2D(target, levels, internalFormat, width, height)
    elif target == GL_TEXTURE_3D or depth > 1:
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER )
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
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    return cap, width, height

def generateTextures(textureDict, numImages, width, height):

    
    maxLevels = 6 # FIXME
    numLevels = 6 # FIXME TOO

    textureDict['lastColor'] = createTexture(textureDict['lastColor'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['nextColor'] = createTexture(textureDict['nextColor'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['lastGradMap'] = createTexture(textureDict['lastGradMap'], GL_TEXTURE_2D, GL_RG32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['nextGradMap'] = createTexture(textureDict['nextGradMap'], GL_TEXTURE_2D, GL_RG32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['lastFlowMap'] = createTexture(textureDict['lastFlowMap'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['nextFlowMap'] = createTexture(textureDict['nextFlowMap'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['sparseFlowMap'] = createTexture(textureDict['sparseFlowMap'], GL_TEXTURE_2D, GL_RGBA32F, maxLevels, int(width / 4), int(height / 4), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['blankFlowMap'] = createTexture(textureDict['blankFlowMap'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['skeletonColor'] = createTexture(textureDict['skeletonColor'], GL_TEXTURE_2D, GL_RGB8, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['depthInColor'] = createTexture(textureDict['depthInColor'], GL_TEXTURE_2D, GL_R16, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

    blankData = np.zeros(int(width*height* 4), dtype='float32')

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['blankFlowMap'])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_RGBA, GL_FLOAT, blankData)
    glGenerateMipmap(GL_TEXTURE_2D)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['lastFlowMap'])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_RGBA, GL_FLOAT, blankData)
    glGenerateMipmap(GL_TEXTURE_2D)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['nextFlowMap'])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_RGBA, GL_FLOAT, blankData)
    glGenerateMipmap(GL_TEXTURE_2D)


	# Allocate the immutable GPU memory storage -more efficient than mutable memory if you are not going to change image size after creation
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, int(width))



    return textureDict




def preprocess(image, mean, std):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def main():


    # initialize glfw
    if not glfw.init():
        return
    #glfw.window_hint(glfw.VISIBLE, False)    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    #creating the window
    window = glfw.create_window(1600, 900, "PyGLFlow", None, None)
    if not window:
        glfw.terminate()
        return

    xPos = 0
    yPos = 0

    maxLevels = 6

    flow_values = deque([0, 0])

    glfw.make_context_current(window)

    kinect()
    import pyk4a



    # segContext = engine.create_execution_context()

    # inputs = np.random.random((512, 320, 3, 1)).astype(np.float32)

    # in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)

    if platform.system() == 'Linux':
        with open('./models/human_pose.json', 'r') as f:
            human_pose = json.load(f)
        
            topology = trt_pose.coco.coco_category_to_topology(human_pose)
            

            WIDTH = 256
            HEIGHT = 256

            #data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

            OPTIMIZED_MODEL = Path('./models/densenet121_baseline_att_256x256_B_epoch_160_trt.pth')

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

            print('loaded model')

            mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
            std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
            device = torch.device('cuda')

            parse_objects = ParseObjects(topology)
            draw_objects = DrawObjects(topology)
    else:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)




    # NAMED TEXTURES
    lastColor = -1            # 0
    nextColor = -1            # 1
    lastGradMap = -1          # 2
    nextGradMap = -1          # 3
    lastFlowMap = -1          # 4
    nextFlowMap = -1          # 5
    sparseFlowMap = -1        # 6
    #densificationFlowMap = -1 # 7
    skeletonColor = -1        # 7



    textureDict = {
        'lastColor' : lastColor, 
        'nextColor' : nextColor, 
        'lastGradMap' : lastGradMap, 
        'nextGradMap' : nextGradMap, 
        'lastFlowMap' : lastFlowMap, 
        'nextFlowMap' : nextFlowMap, 
        'sparseFlowMap' : sparseFlowMap, 
        'skeletonColor' : skeletonColor,
        'blankFlowMap' : -1,
        'depthInColor' : -1,
    }

    bufferDict = {
        'sumFlow' : -1
    }

    bufferDict = generateBuffers(bufferDict)

    densifiactionFBO = -1



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

    inverseSearch_shader = (Path(__file__).parent / 'shaders/disFlow.comp').read_text()

    inverseSearchShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(inverseSearch_shader, GL_COMPUTE_SHADER))

    densifyVert_shader = (Path(__file__).parent / 'shaders/disDensification.vert').read_text()
    densifyFrag_shader = (Path(__file__).parent / 'shaders/disDensification.frag').read_text()

    densifyShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(densifyVert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(densifyFrag_shader, GL_FRAGMENT_SHADER))

    dense_shader = (Path(__file__).parent / 'shaders/disDense.comp').read_text()

    denseShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(dense_shader, GL_COMPUTE_SHADER))

    sum_shader = (Path(__file__).parent / 'shaders/sumFLowROI.comp').read_text()

    sumShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(sum_shader, GL_COMPUTE_SHADER))

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
    depthRange_loc = glGetUniformLocation(shader, "depthRange")


    # make some default background color
    glClearColor(0.2, 0.3, 0.2, 1.0)

    # set some values
    sliderRValue = 0
    sliderGValue = 0
    sliderBValue = 0

    frameCounter = 0

    depthRangeSliderMin = 0.0
    depthRangeSliderMax = 1.0

    #default to not running any filters
    doFilterEnabled = False
    getPose = False
    showFileDialogueOptions = False
    showCameraDialogueOptions = False
    currentFile = 0
    fileList = []
    listy = list(Path('./data/').glob('./*'))
    for x in listy:
        fileList.append(str(x))

    currentCamera = 0
    cameraList = ['0', '1', '2', '3', 'kinect']
    resetVideoSource = True
    sourceAvailable = False

    global cap

    filemode = 0 # 1 : webcam, 2 : video file

    numberOfImages = 5

    width = 0
    height = 0

    numberOfFrames = 1
    firstFrame = True

    #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)

    classColors = np.array([
        [0,   0,   0,],
        [139, 69,  19],
        [222, 184, 135],
        [210, 105, 30],
        [255, 255, 0],
        [255, 165, 0],
        [0  , 255, 0],
        [60 , 179, 113],
        [107, 142, 35],
        [255, 0  , 0],
        [245, 222, 179],
        [0  , 0  , 255],
        [0  , 255, 255],
        [238, 130, 238],
        [128, 0  , 128],
        [255, 0  , 0],
        [255, 0  , 255],
        [128, 128, 128],
        [192, 192, 192],
        [128, 128, 128],
        [128, 128, 128]]
    )


    useKinect = False


    while not glfw.window_should_close(window):

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        if resetVideoSource:
            if sourceAvailable:
                reset()
                if filemode == 1:
                    if (cameraList[currentCamera] == 'kinect'):
                        from pyk4a import Config, PyK4A
                        useKinect = True
                        k4a = PyK4A(
                            Config(
                                color_resolution=pyk4a.ColorResolution.RES_1080P,
                                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                                synchronized_images_only=True,
                            )
                        )
                        k4a.start()
                        capture = k4a.get_capture()
                        width = 1920
                        height = 1080
                    else:
                        cap, width, height = openCamera(cameraList[currentCamera])
                elif filemode == 2:
                    if (fileList[currentFile] == 'data\\vid6.mkv'):
                        from pyk4a import PyK4APlayback
                        useKinect = True
                        playback = PyK4APlayback(fileList[currentFile])
                        playback.open()
                        capture = playback.get_next_capture()
                        if (playback.configuration['color_resolution'] == 2):
                            width = 1920
                            height = 1080
                        if (playback.configuration['depth_mode'] == 2):
                            width_d = 640
                            height_d = 576
                        #cv2.imshow("Color", cv2.imdecode(capture.color, cv2.IMREAD_COLOR))
                        #cv2.waitKey(1)
                        numberOfFrames = playback.length
                    else:
                        cap, width, height = openVideo(fileList[currentFile])
                        numberOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)


                textureDict = generateTextures(textureDict, numberOfImages, width, height)
                #densifiactionFBO = DIS.generateDensificationFramebuffer(textureList[5], width, height)

                
                resetVideoSource = False

        else:
            if not useKinect:
                ret, frame = cap.read()
                blank_frame = np.zeros_like(frame)
            else:
                if filemode == 1:
                    capture = k4a.get_capture()

                    if capture.color is not None:
                        frame = image = cv2.flip(cv2.cvtColor(capture.color, cv2.COLOR_RGBA2RGB ), 0)
                        blank_frame = np.zeros_like(frame)

                        ret = True
                    if capture.transformed_depth is not None:
                            #cv2.imshow("depth", capture.transformed_depth)
                            #cv2.waitKey(1)
                            glActiveTexture(GL_TEXTURE0)
                            glBindTexture(GL_TEXTURE_2D, textureDict['depthInColor'])
                            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_RED, GL_UNSIGNED_SHORT, np.array(capture.transformed_depth))
                    else:
                        ret = False
                elif filemode == 2:
                    try:
                        capture = playback.get_next_capture()
                        if capture.color is not None:
                            frame = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                            frame = cv2.flip(frame, 0)
                            blank_frame = np.zeros_like(frame)
                            ret = True
                        if capture.transformed_depth is not None:
                            #cv2.imshow("depth", capture.transformed_depth)
                            #cv2.waitKey(1)
                            glActiveTexture(GL_TEXTURE0)
                            glBindTexture(GL_TEXTURE_2D, textureDict['depthInColor'])
                            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_RED, GL_UNSIGNED_SHORT, np.array(capture.transformed_depth))
                        else:
                            ret = False
                    except EOFError:
                        break


                

            if imgui.is_mouse_clicked():
                if not imgui.is_any_item_active():
                    mouseX, mouseY = imgui.get_mouse_pos()
                    w, h = glfw.get_framebuffer_size(window)
                    xPos = ((mouseX % int(w / 3)) / (w / 3) * width)
                    yPos = (mouseY / (h)) * height

                    #print(xPos, " ", yPos)

            if ret:

                glBindFramebuffer(GL_FRAMEBUFFER, 0)

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, textureDict['nextColor'])

                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                img_data = np.array(frame.data, np.uint8)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_BGR, GL_UNSIGNED_BYTE, img_data)
                glGenerateMipmap(GL_TEXTURE_2D)


                # inputImage = np.array(cv2.resize(((np.array(frame.data, np.float32) / 255.0) - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225), (320, 512)), dtype=np.float32)

                # res = inference(engine, segContext, inputImage.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
                # otemp = np.zeros((10, 16, 3))

                # s_w = 16
                # s_h = 10
                # s_c = 21

                # for y in range(s_h):
                #     for x in range(s_w):

                #         p_max = -100000.0
                #         c_max = -1
                        
                #         for c in range(1, s_c, 1):

                #             p = res[c * s_w * s_h + y * s_w + x]
                #             if( c_max < 0 or p > p_max ):
                #                 p_max = p
                #                 c_max = c

                #         otemp[y,x,:] = classColors[c_max]

                # #outMask = np.zeros((10, 16)).astype(np.float32)
                # #outy = np.argmax(otemp, axis=0)

                # cv2.imshow('frame', otemp)
                # cv2.resizeWindow('frame', 512,320)
                # cv2.waitKey(1)

                if getPose:
                    if platform.system() == 'Linux':
                        imageSmall = cv2.resize(frame, (width, height))
                        data = preprocess(imageSmall, mean, std)
                        cmap, paf = model_trt(data)
                        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
                        counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
                        draw_objects(blank_frame, counts, objects, peaks)
                    else:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        #image.flags.writeable = True
                        #frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(
                            blank_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        #cv2.imshow("fgr", blank_frame)
                        #cv2.waitKey(1)

                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, textureDict['skeletonColor']) 
                    img_data_pose = np.array(blank_frame.data, np.uint8)

                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_RGB, GL_UNSIGNED_BYTE, img_data_pose)


                    #print(counts)
                    #cv2.imshow('fr', frame)
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, textureDict['blankFlowMap'])
                    img_data = np.array(frame.data, np.uint8)
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(width), int(height), GL_BGR, GL_UNSIGNED_BYTE, img_data)

                if (firstFrame):
                    glCopyImageSubData(textureDict['nextColor'], GL_TEXTURE_2D, 0, 0, 0, 0, textureDict['lastColor'], GL_TEXTURE_2D, 0, 0, 0, 0, int(width), int(height), 1)
                    glBindTexture(GL_TEXTURE_2D, textureDict['lastColor'])
                    glGenerateMipmap(GL_TEXTURE_2D)
                    firstFrame = False


                if (doFilterEnabled):


                    for lvl in range(4, -1, -1):
                        do_gradFilter(gradShader, textureDict, lvl, width, height)
                        do_inverseSearch(inverseSearchShader, textureDict, lvl, width, height)
                        do_densify(denseShader, textureDict, lvl, width, height)
                        #do_densify(densifyShader, densifiactionFBO, textureList, lvl, width, height)


                        currentFlow = do_summing(sumShader, textureDict, bufferDict, lvl, width, height, xPos, yPos)
                        
                        flow_values.appendleft(currentFlow)

                        if len(flow_values) > 1000:
                            flow_values.pop()



   

                w, h = glfw.get_framebuffer_size(window)



                # set the active drawing viewport within the current GLFW window (i.e. we are spliiting it up in 3 cols)
                xpos = 0
                ypos = 0
                xwidth = float(w) / 2.0
                glViewport(int(xpos), int(ypos), int(xwidth),h)
                glClear(GL_COLOR_BUFFER_BIT)

                glUseProgram(shader)


                # glUniform1i(renderType_loc, 0)
                # glUniform1i(sliderR_loc, sliderRValue)

                # #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
                # # DRAW THE FIRST WINDOW (live feed)
                # glActiveTexture(GL_TEXTURE0)
                # if getPose:
                #     glBindTexture(GL_TEXTURE_2D, textureDict['blankFlowMap'])
                # else:
                #     glBindTexture(GL_TEXTURE_2D, textureDict['nextColor'])

                # glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


                # set second draw call's drawing location (we've shifted accros by width / 4)
                xpos = 0                
                glViewport(int(xpos), int(ypos), int(xwidth),h)
                glUniform1i(renderType_loc, 2)

                # we want to now render from the processed texture, whose memory has been populated by the edgeFilter compute shader
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, textureDict['nextFlowMap'])
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                glActiveTexture(GL_TEXTURE4)
                glBindTexture(GL_TEXTURE_2D, textureDict['depthInColor'])
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                glUniform1i(sliderR_loc, sliderRValue)


                # set third draw call's drawing location (we've shifted accros by 2 * width / 3)
                xpos = float(w) / 2.0
                glViewport(int(xpos), int(ypos), int(xwidth),h)
                glUniform1i(renderType_loc, 3)

                glUniform2f(depthRange_loc, depthRangeSliderMin, depthRangeSliderMax)

                glActiveTexture(GL_TEXTURE2)
                glBindTexture(GL_TEXTURE_2D, textureDict['nextGradMap'])

                glActiveTexture(GL_TEXTURE3)
                glBindTexture(GL_TEXTURE_2D, textureDict['skeletonColor'])

                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)


                # set third draw call's drawing location (we've shifted accros by 2 * width / 3)
                # xpos = 3.0 * float(w) / 3.0
                # glViewport(int(xpos), int(ypos), int(xwidth),h)
                # glUniform1i(renderType_loc, 2)

                # glActiveTexture(GL_TEXTURE1)
                # glBindTexture(GL_TEXTURE_2D, textureList[5])

                # glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                # glActiveTexture(GL_TEXTURE1)
                # glBindTexture(GL_TEXTURE_2D, textureList[5])
                # glGenerateMipmap(GL_TEXTURE_2D)

                # swap frame handles

                glCopyImageSubData(textureDict['nextFlowMap'], GL_TEXTURE_2D, 0, 0, 0, 0, textureDict['lastFlowMap'], GL_TEXTURE_2D, 0, 0, 0, 0, int(width), int(height), 1)
                glBindTexture(GL_TEXTURE_2D, textureDict['lastFlowMap'])
                glGenerateMipmap(GL_TEXTURE_2D)

                # glCopyImageSubData(textureList[7], GL_TEXTURE_2D, 0, 0, 0, 0, textureList[5], GL_TEXTURE_2D, 0, 0, 0, 0, width, height, 1)
                # glBindTexture(GL_TEXTURE_2D, textureList[5])
                # glGenerateMipmap(GL_TEXTURE_2D) 

                textureDict['lastColor'], textureDict['nextColor'] = textureDict['nextColor'], textureDict['lastColor']
                #textureList[2], textureList[3] = textureList[3], textureList[2]
               # textureList[4], textureList[5] = textureList[5], textureList[4]

                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, textureDict['lastFlowMap'])
                glGenerateMipmap(GL_TEXTURE_2D)






            elif ret == False and filemode == 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        sTime = time.perf_counter()


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

        changedDepth, depthRangeSliderMin = imgui.slider_float("depthMin", depthRangeSliderMin, min_value=0, max_value=1.0)
        changedDepth, depthRangeSliderMax = imgui.slider_float("depthMax", depthRangeSliderMax, min_value=0, max_value=1.0)

        #changedR, sliderRValue = imgui.slider_int("sliceR", sliderRValue, min_value=0, max_value=5)
        changedG, frameCounter = imgui.slider_int("frame", frameCounter, min_value=0, max_value=numberOfFrames)

        #changedB, sliderBValue = imgui.slider_int("sliceB", sliderBValue, min_value=0, max_value=numberOfImages)
        _, doFilterEnabled = imgui.checkbox("run filter", doFilterEnabled)
        _, getPose = imgui.checkbox("run pose", getPose)

        if changedG:
            if not useKinect:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameCounter)
            else:
                playback.seek(frameCounter)

        imgui.end()

        # meanFlow = np.mean(np.array(flow_values)) # this can cuase crashes .... not sure why
        # stdDevFlow = np.std(np.array(flow_values))
        # smooth_flow_values = uniform_filter1d(flow_values, 30)

        # imgui.begin("plotting")
        # imgui.plot_lines("flow", array('f', smooth_flow_values), 1000, 0, None, meanFlow - (1.5 * stdDevFlow), meanFlow + (1.5 * stdDevFlow), (500, 100))
        # imgui.end()

        imgui.render()

        impl.render(imgui.get_draw_data())



        #print((time.perf_counter() - sTime) * 1000)


        glfw.swap_buffers(window)

        frameCounter = frameCounter + 2

        if frameCounter >= numberOfFrames:
            frameCounter = 0
            if (useKinect and filemode == 2):
                playback.seek(frameCounter)


    glfw.terminate()
    if not useKinect:
        cap.release()
    else:
        playback.close()

if __name__ == "__main__":
    main()
