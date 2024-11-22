import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token
from modelpath import model_path
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
 
device = "cuda"  # Define device globally
 
def initialize():
    global model_id, app, frame, prompt, lmain, pipe, trigger
   
    model_name = input("Enter the model name: ")
    model_id = model_path.get(model_name, None)
   
    if model_id is None:
        print("Model name not found.")
        return
   
    # Create the app
    app = tk.Tk()
    app.geometry("632x832")
    app.title("Dhyey Image Generative App")
    ctk.set_appearance_mode("dark")
 
    # Create a frame to contain the widgets
    frame = tk.Frame(app)
    frame.pack()
 
    # Create the CTkEntry widget with the frame as its master
    prompt = ctk.CTkEntry(frame, height=40, width=512, text_color="black", fg_color="white")
    prompt.pack()
 
    lmain = ctk.CTkLabel(frame, height=712, width=12)
    lmain.pack()
 
    global pipe  # Define pipe globally
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
    pipe.to(device)
 
    trigger = ctk.CTkButton(frame, height=40, width=120, text_color="white", fg_color="blue", command=generate)
    trigger.configure(text="Generate")
    trigger.pack()
 
def generate():
    global device  # Access device globally
    with autocast(device):
        output = pipe(prompt.get(), guidance_scale=8.5)
        image = output["images"][0]
 
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img
 
if __name__ == "__main__":
    initialize()
    app.mainloop()