import argparse
import os
from random import random, randint, sample

import numpy as np
import oneflow.experimental as flow
import oneflow.experimental.nn as nn

from models.deep_q_network import DeepQNetwork
from utils.flappy_bird import FlappyBird
from utils.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()

    model = DeepQNetwork()
    model.to('cuda')

    optimizer = flow.optim.Adam(model.parameters(), lr=opt.lr)

    criterion = nn.MSELoss()
    criterion.to('cuda')

    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)

    image = flow.Tensor(image, dtype=flow.float32)
    image = image.to('cuda')

    state = flow.cat(tuple(image for _ in range(4))).unsqueeze(0)

    replay_memory = []
    iter = 0
    # weights_list = []
    while iter < opt.num_iters:
        model.train()
        prediction = model(state)[0]
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:
            action = flow.argmax(prediction).numpy()[0]
            print("action: ", action)

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)

        next_image = flow.Tensor(next_image)
        next_image = next_image.to('cuda')

        next_state = flow.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)

        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = flow.cat(tuple(state for state in state_batch))
        action_batch = flow.Tensor(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = flow.Tensor(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = flow.cat(tuple(state for state in next_state_batch))

        state_batch = state_batch.to('cuda')
        action_batch = action_batch.to('cuda')
        reward_batch = reward_batch.to('cuda')
        next_state_batch = next_state_batch.to('cuda')

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = flow.cat(
            tuple(reward_batch[i] if terminal_batch[i] else reward_batch[i] + opt.gamma * flow.max(next_prediction_batch[i]) for i in
                  range(len(reward_batch.numpy()))))

        q_value = flow.sum(current_prediction_batch * action_batch, dim=1)

        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        state = next_state
        iter += 1

        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            opt.num_iters,
            action,
            loss.numpy()[0],
            epsilon, reward, flow.max(prediction).numpy()[0]))

        if (iter+1) % 100000 == 0:
            flow.save(model.state_dict(), os.path.join(opt.saved_path, "epoch_%d" % (iter+1)))
    flow.save(model.state_dict(), os.path.join(opt.saved_path, "final_models"))
    print("train success!")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
