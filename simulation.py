if __name__ == "__main__":
    import gym
    env = gym.make('yumi-v0')
    observation = env.reset()
    env.render()

    while True:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
