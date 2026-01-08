import modal

app = modal.App("jpeg-test")
vol = modal.Volume.from_name("ironclad-test-vol")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch", "torchvision", "diffusers", "transformers", "open_clip_torch", "pillow"
)

@app.function(image=image, volumes={"/vol": vol}, gpu="T4", timeout=300)
def test_jpeg_robustness():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    from diffusers import AutoencoderKL
    import open_clip

    device = "cuda"
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to(device)
    vae.eval()
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device).eval()

    transform = T.ToTensor()

    # train_normalの最初の画像（sap_v3_variantsで使用されるのと同じ）
    from pathlib import Path
    train_normal = Path("/vol/train_normal")
    image_files = sorted(list(train_normal.glob("*.png")) + list(train_normal.glob("*.jpg")))
    orig_img = Image.open(image_files[0]).convert("RGB")
    orig_tensor = transform(orig_img).unsqueeze(0).to(device)

    with torch.no_grad():
        z_orig = vae.encode(orig_tensor * 2 - 1).latent_dist.mean
        orig_clip = F.interpolate(orig_tensor, size=(224, 224), mode='bilinear')
        orig_clip_norm = (orig_clip - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)) / \
                         torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        clip_orig = clip_model.encode_image(orig_clip_norm)
        clip_orig = F.normalize(clip_orig, dim=-1)

    # 保護済みPNGのlatentを基準に（圧縮前後の比較）
    protected_img = Image.open("/vol/test_sap_v3_perlin_scale64.png").convert("RGB")
    protected_tensor = transform(protected_img).unsqueeze(0).to(device)
    with torch.no_grad():
        z_protected = vae.encode(protected_tensor * 2 - 1).latent_dist.mean

    test_images = [
        ("Original (未保護)", "/vol/test_sap_v3_perlin_original.png"),
        ("Protected PNG", "/vol/test_sap_v3_perlin_scale64.png"),
        ("JPEG Q=90", "/vol/test_sap_v3_jpeg_q90.jpg"),
        ("JPEG Q=80", "/vol/test_sap_v3_jpeg_q80.jpg"),
        ("JPEG Q=70", "/vol/test_sap_v3_jpeg_q70.jpg"),
    ]

    results = []
    for name, path in test_images:
        img = Image.open(path).convert("RGB")
        if img.size != orig_img.size:
            img = img.resize(orig_img.size, Image.LANCZOS)
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z = vae.encode(tensor * 2 - 1).latent_dist.mean
            vae_sim = F.cosine_similarity(z_orig.flatten(), z.flatten(), dim=0).item()
            
            clip_input = F.interpolate(tensor, size=(224, 224), mode='bilinear')
            clip_norm = (clip_input - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)) / \
                        torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
            clip_feat = clip_model.encode_image(clip_norm)
            clip_feat = F.normalize(clip_feat, dim=-1)
            clip_sim = F.cosine_similarity(clip_orig, clip_feat).item()
        
        results.append((name, vae_sim, clip_sim))
    
    return results

@app.local_entrypoint()
def main():
    results = test_jpeg_robustness.remote()
    print("\n" + "="*60)
    print("JPEG圧縮耐性テスト")
    print("="*60)
    print(f"{'Image':<20} {'VAE Cos Sim':>12} {'CLIP to Orig':>12}")
    print("-"*60)
    for name, vae, clip in results:
        print(f"{name:<20} {vae:>12.4f} {clip:>12.4f}")
    print("="*60)
