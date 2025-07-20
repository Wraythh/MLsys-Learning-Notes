import asyncio
import threading
import time

class Trainer:
    def __init__(self):
        self.rollout_wg = [0, 0]
        self.train_wg = [0, 0]
        self.buffer = None

    async def async_generate_sequence(self):
        while True:
            if self.buffer.full():
                await asyncio.sleep(0.001)
                continue
            print("generating...")
            await asyncio.sleep(0.2)
            data = [x + 1 for x in self.rollout_wg]
            print(f"finish generate :{data}")
            await self.buffer.put(data)

    async def update_weights(self):
        print("updating weights")
        self.rollout_wg = self.train_wg

    async def train(self):
        while True:
            data = await self.buffer.get()
            print("training...")
            print(f"start train :{data}")
            await asyncio.sleep(1)
            data = [x + 1 for x in data]
            self.train_wg = data
            await self.update_weights()
            print(f"finish train :{data}")

    async def fit(self):
        self.buffer = asyncio.Queue(1)
        generate_task = asyncio.create_task(self.async_generate_sequence())
        train_task = asyncio.create_task(self.train())
        await asyncio.gather(generate_task, train_task)

def main():
    async_trainer = Trainer()
    asyncio.run(async_trainer.fit())

if __name__ == "__main__":
    main()