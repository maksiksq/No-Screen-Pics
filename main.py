import os

import numpy as np
import aiohttp
import discord
import aiofiles
import dotenv
import json
import PIL

def load_switches():
    with open("switches.json") as f:
        return json.load(f)

def save_switches(data):
    with open("switches.json", "w") as f:
        json.dump(data, f)

from tensorflow import keras

guild_switches = {}

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

@bot.slash_command(name="verbose", description="Adds accuracy numbers to bot's responses")
async def verbose(ctx):
    if not ctx.author.guild_permissions.administrator:
        await ctx.respond("Nuh uh. You must be an admin to use this command. ", ephemeral=True)
        return
    guild_id = ctx.guild_id
    current = guild_switches.get(guild_id, False)
    guild_switches[guild_id] = not current

    save_switches(guild_switches)

    print(guild_switches[guild_id])
    await ctx.respond(f"Verbose mode is now set to {guild_switches[guild_id]}. Run the command again to toggle.")


@bot.event
async def on_message(message):
    if not message.attachments:
        return

    print("receiving message")
    print(message.attachments[0].url)
    if message.author == bot.user:
        return

    for attachment in message.attachments:
        # await message.channel.send('Scanning beep boop')

        img_bytes = await fetch_image(attachment.url)
        confidence = await classify_image_w_bytes(img_bytes)

        guild_id = message.guild.id

        verbosnt = guild_switches.get(guild_id)
        if verbosnt is None:
            guild_switches[guild_id] = False
            verbosnt = guild_switches.get(guild_id, False)
        print(f"Verbose: {verbosnt}")

        if confidence < 0.3:
            if verbosnt:
                await message.reply(f"Prediction (0 = bad photo, 1 = screenshot): {confidence}")
            await message.reply("**Viewing a photo of a screen is not very pleasant, particularly if there's text to read. Simply take a screenshot instead, it's faster and easier.**\nHow to take a screenshot:\n- within Minecraft: F2 \n- Windows: Win + Shift + S \n- Mac: Shift + Command + 4")

            # old response
            # await message.reply("❌ PLEASE, FOR GOD'S SAKE, JUST MAKE A SCREENSHOT.")

            print(f"new message:\n {attachment.url}")
            print("❌ PLEASE, FOR GOD'S SAKE, JUST MAKE A SCREENSHOT.")

        else:
            if verbosnt:
                await message.reply(f"Prediction (0 = bad photo, 1 = screenshot): {confidence}")
                await message.reply("✅ This is a proper screenshot. Doing nothing.")

            # Doing nothing
            print(f"new message:\n {attachment.url}")
            print("✅ This is a proper screenshot, doing nothing.")


token = str(os.getenv("TOKEN"))

bot.run(token)
