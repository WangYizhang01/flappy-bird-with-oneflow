import argparse
import os
import oneflow
import oneflow.experimental as flow
import oneflow.experimental.nn as nn

from models.deep_q_network import DeepQNetwork
from utils.flappy_bird import FlappyBird
from utils.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt):
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()

    model = DeepQNetwork()
    model.eval()
    model.to('cuda')
    model.load_state_dict(flow.load(os.path.join(opt.saved_path, "epoch_%d" % 2000000)))

    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = flow.Tensor(image)
    image = image.to('cuda')
    state = flow.cat(tuple(image for _ in range(4))).unsqueeze(0)

    while True:
        prediction = model(state)[0]
        action = flow.argmax(prediction).numpy()[0]
        
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = flow.Tensor(next_image)
        next_image = next_image.to('cuda')
        next_state = flow.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)

        state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt)
