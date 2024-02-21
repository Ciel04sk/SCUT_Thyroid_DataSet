import setup_path
import cv2
import gym
import numpy as np
import os

from gym import spaces
from thyroid_glanod_gym.envs.thyroid_glanod_env import Thyroid_GlaNod_Env
from PIL import Image
from thyroid_glanod_gym.envs.cenvWrap import Cenv
import copy
cv2.namedWindow("bimg")

class ThyroidGlaNodDiscreteCJPEnv(Thyroid_GlaNod_Env):
    cenv=Cenv()
    shStpCnt=0
    def __init__(self, image_shape):
        self.shStpCnt=0
        self.cenv.addNoise(r"panoramaSim\samples\noise3", "*.png")
        
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\1", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\2", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\3", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\4", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\5", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\6", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\7", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\8", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\9", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\10", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\11", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\12", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\CJP_edit3\13", "*.png", True, True)


        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\1", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\2", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\3", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\4", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\5", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\6", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\7", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\8", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\9", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\10", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\11", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\12", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\13", "*.png", True, True)
        self.cenv.addSeq(r"panoramaSim\samples\HYX_edit3\14", "*.png", True, True)
        super().__init__(image_shape)
        
        self.image_shape = image_shape
        self.img_width, self.img_height, self.channel = self.image_shape
        self.action_space = spaces.Discrete(4)
        
        self.eps_rew = 0
        self.best_rew = -1000
        
        self.eps_count = 0
        self.best_rew_count = 0

    def __del__(self):
        pass
        
    def reset(self):
        self.cenv.reInit()
        self.eps_rew = 0
        
        return self._get_obs()
    
    def step(self, action):
        if(action!=3):
            obs,reward,done,realDst,realConf =self.cenv.step(action-1,1)
        else:
            obs,reward,done,realDst,realConf =self.cenv.step(0,0)
        
        self.eps_rew += reward

        if done == 1:
            self.eps_count += 1
            
            if (self.eps_rew > self.best_rew):
                self.best_rew_count += 1
                self.best_rew = self.eps_rew
        
        obs=np.array(Image.fromarray(obs).resize((64,64)))
        
        cv2.imshow('bimg', obs)
        if(self.shStpCnt>1):
            self.shStpCnt=0
            actStr=['left','mid','right','err']
            print(f"ep_c:{self.eps_count}, act:{actStr[action]}, rew:{reward:.2f}, ep_rew:{self.eps_rew:.2f}, best_rew_c:{self.best_rew_count}, best_rew:{self.best_rew:.2f}")
        self.shStpCnt+=1
        cv2.waitKey(2)
        
        obs=obs.reshape(64,64,1)
      
        return obs, reward, done, {}
    
    """--------Step 2--------"""
    def _get_obs(self):
        obs,reward,done,_=self.step(0)
        return obs

