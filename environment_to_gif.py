from PIL import Image, ImageDraw
import gym
from pettingzoo.mpe import simple_spread_v2
from matplotlib import animation
import matplotlib.pyplot as plt

def ellipse(x, y, offset):
    image = Image.new("RGB", (400, 400), "blue")
    draw = ImageDraw.Draw(image)
    draw.ellipse((x, y, x+offset, y+offset), fill="red")
    return image

def make_gif():
    frames = []
    x = 0
    y = 0
    offset = 50
    for number in range(20):
        frames.append(ellipse(x, y, offset))
        x += 35
        y += 35
        
    frame_one = frames[0]
    frame_one.save("circle.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)

def sample_action(env, agents):
    action = {}
    for agent in agents:
        action_space = env.action_space(agent)
        action[agent] = action_space.sample()
    return action

# code from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


if __name__ == "__main__":
    env_name = 'simple_spread_v2'

    env_config = {
        'N':3, 
        'local_ratio': 0.5, 
        'max_cycles': 100, 
        'continuous_actions': True
    }

    env = simple_spread_v2.parallel_env(N = env_config['N'], 
                    local_ratio = env_config['local_ratio'], 
                    max_cycles = env_config['max_cycles'], 
                    continuous_actions = env_config['continuous_actions'])    
    
    
    state_dict = env.reset()
    print(state_dict)
    agents = env.agents

    frames = []

    for i in range(100):
        env.render()
        frames.append(env.render(mode="rgb_array"))
        action_dict = sample_action(env, agents)  
        # print("action", action_dict)
        obs, reward, terminated, info = env.step(action_dict)
        print(obs)
        # obs, reward, terminated, info
        # print(obs)

    env.close()
    print(len(frames))
    save_frames_as_gif(frames)
    # make_gif()