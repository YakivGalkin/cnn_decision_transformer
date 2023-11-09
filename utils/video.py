import imageio
import numpy as np
import os
from base64 import b64encode
import gymnasium as gym

from utils.env_driver import EnvDriverBase

def record_drive_video(env_driver : EnvDriverBase, file_name, max_ep_len=999):
    env =  gym.make('CarRacing-v2', render_mode="rgb_array", continuous=False)
    frames = []
    [state, _] = env.reset()
    episode_return, episode_length, reward = 0, 0, 0
    for t in range(max_ep_len):
        action = env_driver.drive(state, previous_reward = reward)
        state, reward, done, _, _ = env.step(action)
        episode_return += reward
        frames.append(env.render())
        if done:
            break
    return record_video(file_name, frames)


def record_video(file_name, frames):
    output_folder = './recorded_videos'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, file_name)
    if os.path.exists(output_path):
        os.remove(output_path)

    with imageio.get_writer(output_path, fps=30, quality=5, codec='libx264') as writer:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            writer.append_data(frame)
    writer.close()
    print(f"Video saved to {output_path}")
    return output_path

def get_video_tag(file_path):
    video_encoded = b64encode(open(file_path, "rb").read()).decode()
    video_tag = f'<video width="640" height="480" controls><source src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>'
    return video_tag

