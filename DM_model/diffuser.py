
#!pip install diffusers
#!pip install accelerate

from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

letter = 'C'

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
# ddpm.to("cuda")
# run pipeline in inference (sample random noise and denoise)
tot=0
while tot<10000:
  image = ddpm(batch_size=100).images
  for ix, ig in enumerate(image):
      # save image
      ig.save("./DataDM/"+letter+str(ix+tot)+".png")
  tot+=100