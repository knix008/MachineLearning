---
language:
- en
license: other
license_name: flux-non-commercial-license
extra_gated_prompt: >-
  By clicking "Agree", you agree to the [FLUX Non-Commercial License
  Agreement](LICENSE.md)
  and acknowledge the [Acceptable Use
  Policy](https://bfl.ai/legal/usage-policy).
tags:
- image-generation
- image-editing
- flux
- diffusion-single-file
pipeline_tag: image-to-image
library_name: diffusers
---

![Teaser](./realism.jpg)
![Teaser](./editing.jpg)
![Teaser](./others.jpg)

The FLUX.2 [klein] model family are our fastest image models to date. FLUX.2 [klein] unifies generation and editing in a single compact architecture, **delivering state-of-the-art quality with end-to-end inference in as low as under a second**. Built for applications that require real-time image generation without sacrificing quality.

FLUX.2 [klein] 9B is a 9 billion parameter rectified flow transformer capable of generating images from text descriptions and supports multi-reference editing capabilities.

Our flagship small model. Defines the Pareto frontier for quality vs. latency across text-to-image, single-reference editing, and multi-reference generation. Matches or exceeds models 5x its size—in under half a second. Built on a 9B flow model with 8B Qwen3 text embedder, step-distilled to 4 inference steps.

This repository holds an FP8 version of FLUX.2 [klein] 9B. The main repository of this model (full BF16 weights) can be found [here](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).

For more information, please read our [blog post](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence).

# **Key Features**

1. A distilled model for sub-second image generation with outstanding quality.
2. Text-to-image and image-to-image multi-reference editing in a single unified model.
3. Great for real-time generation and integration into applications.
4. Ideal for creative exploration with excellent prompt adherence and output diversity.
5. Available for non-commercial use.

# **Usage**

We provide a reference implementation of FLUX.2 [klein] 9B, as well as sampling code, in a dedicated [GitHub repository](https://github.com/black-forest-labs/flux2). Developers and creatives looking to build on top of FLUX.2 [klein] 9B are encouraged to use this as a starting point.

## **API Endpoints**

The FLUX.2 [klein] 9B model is available via the BFL API:

- [bfl.ai](https://bfl.ai)

FLUX.2 [klein] 9B is also available in both [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Diffusers](https://github.com/huggingface/diffusers).

## **Using with Diffusers 🧨**

To use FLUX.2 [klein] 9B with the 🧨 Diffusers python library, first install or upgrade diffusers:

```shell
pip install -U diffusers
```
Then you can use Flux2KleinPipeline to run the model:
```python
import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]
image.save("flux-klein.png")
```

---
Limitations

- This model is not intended or able to provide factual information.
- While the model can output text, text rendered may be inaccurate or subject to distortion.
- As a statistical model, this checkpoint may represent or amplify biases observed in the training data.
- The model may fail to generate output that matches the prompts.
- Prompt following is heavily influenced by the prompting style.

Out-of-Scope Use

The model and its derivatives may not be used:

- In any way that violates applicable law.
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way; including but not limited to the solicitation, creation, acquisition, or dissemination of child exploitative content.
- To generate or disseminate deceptive, fraudulent, misleading or otherwise harmful content.
- To generate or disseminate personal identifiable information that can be used to harm an individual.
- To harass, abuse, threaten, stalk, or bully individuals or groups of individuals.
- To create non-consensual intimate imagery or illegal pornographic content.
- For fully automated decision making or high risk applications that adversely impact an individual's legal rights or otherwise create or modify a binding, enforceable obligation.

Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.

Hardware

The FLUX.2 [klein] 9B model fits in ~29GB VRAM and is accessible on NVIDIA RTX 4090 and above.

---
Responsible AI Development

Black Forest Labs is committed to the responsible development and deployment of our models. Prior to releasing the FLUX.2 family of models, we evaluated and mitigated a number of risks in our model checkpoints and hosted services, including the generation of unlawful content, including child sexual abuse material (CSAM) and nonconsensual intimate imagery (NCII). We implemented a series of pre-release mitigations to help prevent misuse by third parties, with additional post-release mitigations to help address residual risks:

1. Pre-training mitigation. We filtered pre-training data for multiple categories of "not safe for work" (NSFW) and known child sexual abuse material (CSAM) to help prevent a user generating unlawful content in response to text prompts or uploaded images. We have partnered with the https://www.iwf.org.uk/, an independent nonprofit organization dedicated to preventing online abuse, to filter known CSAM from the training data.
2. Post-training mitigation. Subsequently, we undertook multiple rounds of targeted fine-tuning to provide additional mitigation against potential abuse, including both text-to-image (T2I) and image-to-image (I2I) attacks. By inhibiting certain behaviors and suppressing certain concepts in the trained model, these techniques can help to prevent a user generating synthetic CSAM or NCII from a text prompt, or transforming an uploaded image into synthetic CSAM or NCII.
3. Ongoing evaluation. Throughout this process, we conducted multiple internal and external third-party evaluations of model checkpoints to identify further opportunities for mitigation. External third-party evaluations focused on eliciting CSAM and NCII through adversarial testing with (i) text-only prompts, (ii) a single uploaded reference image with text prompts, and (iii) multiple uploaded reference images with text prompts. Based on this feedback, we conducted further safety fine-tuning to produce our open-weight FLUX.2 [klein] models.
4. Release decision. After safety fine-tuning and prior to release, we conducted a final third-party evaluation of the proposed release checkpoints, focused on T2I and I2I generation of synthetic CSAM and NCII, including a comparison with other open-weight T2I and I2I models. The final FLUX.2 [klein] checkpoints demonstrated high resilience against violative inputs in complex generation and editing tasks, and demonstrated higher resilience than leading open-weight models across these risk categories. Based on these findings, we approved the release of the open-weight FLUX.2 [klein] 4B models under an Apache 2.0 license and the release of the FLUX.2 [klein] 9B models under a non-commercial license to support third-party research and development.
5. Inference filters. The repository for the FLUX.2 [klein] models includes filters for NSFW and protected content in inputs and outputs. Filters or manual review must be used with the FLUX.2 [klein] 9B models under the terms of the FLUX Non-Commercial License, and we encourage deployers to implement these mitigations when using the FLUX.2 [klein] 4B models. Where we implement these features on our own hosted services, we may apply multiple filters to intercept text prompts, uploaded images, and output images. We utilize both in-house and third-party filters to mitigate against harmful outputs, such as CSAM and NCII outputs, including filters provided by https://thehive.ai/ and https://www.microsoft.com/.
6. Content provenance. Content provenance features can help users and platforms better identify, label, and interpret AI-generated content online. The inference code for FLUX.2 [klein] implements an example of pixel-layer watermarking. Additionally, this repository includes links to the https://c2pa.org/ standard for metadata. The API for FLUX.2 [klein] applies cryptographically-signed C2PA metadata to downloaded output content to indicate that images were produced with our model.
7. Policies. Acceptable use of our models and access to our API are governed by policies set out in applicable documentation, including FLUX Non-Commercial License (for our non-commercial open-weight users); Developer Terms of Service, Self-Hosted Commercial License Terms, and Usage Policy (for our commercial open-weight model users); and Developer Terms of Service, FLUX API Service Terms, and Usage Policy (for our API users). These prohibit the generation of unlawful content or the use of generated content for unlawful, defamatory, or abusive purposes. Developers and users must consent to these conditions to access the FLUX.2 [klein] 9B models on Hugging Face.
8. Safety. Black Forest Labs takes model safety seriously. We may monitor use to detect misuse or abuse of our models and services. We provide a dedicated email address (safety@blackforestlabs.ai) to solicit feedback from the community. We maintain a reporting relationship with organizations such as the https://www.iwf.org.uk/ and the https://www.missingkids.org/, and welcome ongoing engagement with authorities, developers, and researchers to share intelligence about emerging risks and develop effective mitigations.

---
License

This model falls under the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md).

Trademarks & IP

This project may contain trademarks or logos for projects, products, or services. Use of Black Forest Labs and FLUX trademarks or logos in modified versions of this project must not cause confusion or imply sponsorship or endorsement. Any use of third-party trademarks, intellectual property or logos are subject to those third-party's policies.
