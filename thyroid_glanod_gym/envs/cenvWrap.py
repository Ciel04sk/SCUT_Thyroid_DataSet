from ctypes import *
import numpy as np
class Cenv:
    dll=None
    envRef=None

    def __init__(self):
        self.dll = cdll.LoadLibrary(r'D:\HYX\panoramaSim\x64\DebugFast\panoramaSim.dll')
        self.envRef:c_uint32=self.dll.createEnv()        
    def addSeq(self,path,suffix,canSkipRecal,flip):
        self.dll.addSeq(self.envRef,cast(create_string_buffer(path.encode('gbk')),POINTER(c_char)),cast(create_string_buffer(suffix.encode('gbk')),POINTER(c_char)),c_bool(canSkipRecal),c_bool(flip))
        
    def addNoise(self,path,suffix):
        self.dll.addNoise(self.envRef,cast(create_string_buffer(path.encode('gbk')),POINTER(c_char)),cast(create_string_buffer(suffix.encode('gbk')),POINTER(c_char)))
        
    
    def step(self,ofs,certainty):
        ret=np.zeros((132,128)).astype(np.uint8)       
        reward=np.zeros(1).astype(np.float32)
        finished=np.zeros(1).astype(np.bool_)
        realDst=np.zeros(1).astype(np.float32)
        confidence=np.zeros(1).astype(np.float32)
        
        reward_ptr=cast(reward.ctypes.data,POINTER(c_float))
        finished_ptr=cast(finished.ctypes.data,POINTER(c_bool))
        ret_ptr=cast(ret.ctypes.data,POINTER(c_uint8))
        realDst_ptr=cast(realDst.ctypes.data,POINTER(c_float))
        confidence_ptr=cast(confidence.ctypes.data,POINTER(c_float))
        
        self.dll.step(self.envRef,c_float(ofs),c_float(certainty),ret_ptr,reward_ptr ,finished_ptr,realDst_ptr,confidence_ptr)
     
        #print("a",self.realDst)
        return ret,reward[0],finished[0],realDst[0],confidence[0]
    
    def reInit(self):
         self.dll.reInit(self.envRef)

    
