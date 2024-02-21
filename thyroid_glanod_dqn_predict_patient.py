import gym
import ray
import csv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from recvUS import *
from torchvision import transforms
from PIL import Image

# Load the saved model
model = DQN.load(r"model\thyroid_glanod_dqn_cjp_new2_model\best_model.zip")

DATASET_DIRPATH = r"test_data"
DATASET_FOLDERS = [item for item in os.listdir(DATASET_DIRPATH) if os.path.isdir(os.path.join(DATASET_DIRPATH, item))]
# DATASET_FOLDERS = ["20230722-2217"]


for folder in DATASET_FOLDERS:
    IMG_DIR = os.path.join(DATASET_DIRPATH, folder, "img")
    img_list = os.listdir(IMG_DIR)
    saved_csv_path = os.path.join(DATASET_DIRPATH, folder, folder+"_predict.csv")
    
    with open(saved_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "action", "state"])
        

        for filename in img_list:
            name = filename.split(".")[0]
            us_img = Image.open(os.path.join(IMG_DIR, filename)).convert("L")
            us_img_arr = np.array(us_img.resize((64,64)))
            action, state = model.predict(us_img_arr.reshape(64,64,1), deterministic=True)
            writer.writerow([name, action, state])
            print(f"act_{name}: {action}")


# while not done:
#     count += 1
#     ultrasound_img, entropy  = ray.get(us.getImg.remote())
    
#     if isinstance(ultrasound_img, Image.Image):
#         mini_scanned_img = np.array(ultrasound_img.resize((64,64)))
#         cv2.imshow('scanned_img', mini_scanned_img)
#         action, state = model.predict(mini_scanned_img.reshape(64,64,1), deterministic=False)
#         # Perform necessary actions with the obtained action, observation, reward, etc.
#         print(f"action_{count}: {action}")
#         cv2.waitKey(1)
#     sleep(0.08)