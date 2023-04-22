from diffusers import StableDiffusionPipeline
import torch, os


def main():
    device = "cuda"
    promptList = ["adenocarcinoma",
                  "normal"]
    generator = torch.Generator(device="cuda").manual_seed(466)

    os.system("mkdir output/LUAD/images/")
    # check points
    for checkpoint in [name for name in os.listdir("output/LUAD") if "checkpoint"in name]:
        model_path = os.path.join("output/LUAD/", checkpoint, "pytorch_model.bin")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe.unet.load_attn_procs(model_path)
        pipe.to(device)

        os.system("mkdir " + os.path.join("output/LUAD/images/", checkpoint))
        for prompt in promptList:
            text = "H&E stain pathology image with lung" + prompt
            os.system("mkdir " + os.path.join("output/LUAD/images/", checkpoint, prompt))
            for i in range(8):
                image = pipe(text, num_inference_steps=30, generator=generator).images[0]
                image.save(os.path.join("output/LUAD/images/", checkpoint, prompt, "image%d.png"%i))

    # original
    checkpoint = "checkpoint-0"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.to(device)

    os.system("mkdir " + os.path.join("output/LUAD/images/", checkpoint))
    for prompt in promptList:
        text = "H&E stain pathology image with lung" + prompt
        os.system("mkdir " + os.path.join("output/LUAD/images/", checkpoint, prompt))
        for i in range(8):
            image = pipe(text, num_inference_steps=30, generator=generator).images[0]
            image.save(os.path.join("output/LUAD/images/", checkpoint, prompt, "image%d.png"%i))

    # last
    checkpoint = "checkpoint-last"
    model_path = os.path.join("output/LUAD/", "pytorch_lora_weights.bin")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.unet.load_attn_procs(model_path)
    pipe.to(device)

    os.system("mkdir " + os.path.join("output/LUAD/images/", checkpoint))
    for prompt in promptList:
        text = "H&E stain pathology image with lung" + prompt
        os.system("mkdir " + os.path.join("output/LUAD/images/", checkpoint, prompt))
        for i in range(8):
            image = pipe(text, num_inference_steps=30, generator=generator).images[0]
            image.save(os.path.join("output/LUAD/images/", checkpoint, prompt, "image%d.png"%i))

if __name__ == "__main__":
    main()