# A Fully Autonomous Robotic Ultrasound System for Thyroid Scanning
![Image text](images/1.png)
This code accompanies the paper "A Fully Autonomous Robotic Ultrasound System for Thyroid Scanning".
# Abstract:
The current thyroid ultrasound relies heavily on the experience and skills of the sonographer and the expertise of the radiologist, 
and the process is physically and cognitively exhausting. In this paper, we report a novel fully autonomous robotic ultrasound system (FARUS),
which is able to scan thyroid regions without human assistance and identify malignant nodules. In this system, human skeleton point recognition, 
reinforcement learning, and force feedback are used to deal with the difficulties in locating thyroid targets. The orientation of the ultrasound 
probe is adjusted dynamically via Bayesian optimization. Experimental results on volunteering participants demonstrated that FARUS can perform 
high-quality ultrasound scans, close to manual scans obtained by clinicians, may detect thyroid nodules and can provide data on nodule 
characteristics for ACR TI-RADS calculation.

## 1. System Requirments
Python 3.8.15

## 2. Install Guide
This code requirements are stated at requirements.txt

## 3. Demo
![Image text](2.jpg)
### 1. Dataset introduction
With the approval of the ethical review committee, we recruited of several groups of volunteers. Firstly, we employed FARUS system to autonomously scan 
70 volunteers (SCUTG6K). To address the limitations of handheld US equipment in accurately diagnosing nodules, we opted to employ portable US equipment 
to gather additional sets of data (SCUTG2K).
#### Thyroid Dataset(SCUTG8K)
SCUTG8K can be obtained via https://drive.google.com/drive/folders/1z-n69dk_ANT3ZstAhpjBA2SFz2DrxEmM?usp=drive_link
## 4. Instructions for User
#### RL demo
dependecy requirements:
gym: v_0.21.0
stable_baselines3: v
ray



Training:
run thyroid_glanod_dqn_cjp_train.py file

Predict:
run thyroid_glanod_dqn_predict_patient.py file
## 5. Citations
If you find our work useful or our work gives you any insights, please cite:
Su, K., Liu, J., Ren, X. et al. A fully autonomous robotic ultrasound system for thyroid scanning. Nat Commun 15, 4004 (2024). https://doi.org/10.1038/s41467-024-48421-y
## 6. Lisence
MIT License

Copyright (c) 2023 Kang Su

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




