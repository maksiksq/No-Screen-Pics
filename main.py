import os

import numpy as np
import aiohttp
import discord
import aiofiles
import dotenv
import PIL

from tensorflow import keras

dotenv.load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Bot(intents=intents)

model = keras.models.load_model("screen_photo_detector.keras")


async def classify_image(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    print(f" Prediction: {prediction[0][0]}")

    return prediction[0][0]


async def fetch_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()


async def classify_image_w_bytes(image_bytes):
    async with aiofiles.tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        await temp_file.write(image_bytes)

    result = await classify_image(temp_path)

    return result


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")


@bot.slash_command(name="hello", description="Say hello to the bot")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")


@bot.event
async def on_message(message):
    if not message.attachments:
        return

    print("receiving message")
    print(message.attachments[0].url)
    if message.author == bot.user:
        return

    for attachment in message.attachments:
        await message.channel.send('Scanning beep boop')

        img_bytes = await fetch_image(attachment.url)
        confidence = await classify_image_w_bytes(img_bytes)

        if confidence < 0.5:
            await message.channel.send(f"Prediction (0 = bad photo, 1 = screenshot): {confidence}")
            await message.channel.send("❌ PLEASE, FOR GOD'S SAKE, JUST MAKE A SCREENSHOT.")
            print(f"new message:\n {attachment.url}")
            print("❌ PLEASE, FOR GOD'S SAKE, JUST MAKE A SCREENSHOT.")

        else:
            await message.channel.send(f"Prediction (0 = bad photo, 1 = screenshot): {confidence}")
            await message.channel.send("✅ This is a proper screenshot. Doing nothing.")
            print(f"new message:\n {attachment.url}")
            print("✅ This is a proper screenshot, doing nothing.")


token = str(os.getenv("TOKEN"))

bot.run(token)
